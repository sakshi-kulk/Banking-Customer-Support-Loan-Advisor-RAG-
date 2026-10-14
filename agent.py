"""LangGraph banking workflow and optional LangChain tool-calling agent."""

import re

from state import BankingState
from tools.document_search_tool import build_document_search_tool
from tools.eligibility_tool import build_eligibility_tool, check_eligibility
from tools.emi_tool import build_emi_tool, calculate_emi
from tools.faq_tool import build_faq_tool


def classify_intent(question: str) -> str:
    text = question.lower()
    if re.search(r"\b(eligible|eligibility|qualify|qualifies)\b", text):
        return "eligibility"
    if re.search(r"\bemi\b|monthly installment|monthly instalment", text):
        return "emi"
    return "faq"


def _amount_to_inr(value: str, unit: str | None) -> float:
    amount = float(value.replace(",", ""))
    if unit and unit.lower().startswith(("lakh", "lac")):
        amount *= 100_000
    elif unit and unit.lower().startswith(("crore", "cr")):
        amount *= 10_000_000
    elif unit and unit.lower() == "k":
        amount *= 1_000
    return amount


def parse_emi_question(question: str) -> dict[str, float]:
    text = question.lower()
    rate_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    tenure_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", text)
    if not rate_match or not tenure_match:
        raise ValueError("Include an annual interest rate (for example 9%) and tenure in years.")

    amount_pattern = re.compile(
        r"(?:₹|rs\.?\s*|rupees?\s*)?\s*(\d[\d,]*(?:\.\d+)?)\s*"
        r"(lakhs?|lacs?|crores?|cr|k)?\b",
        re.IGNORECASE,
    )
    principal = None
    for match in amount_pattern.finditer(text):
        start, end = match.span()
        if start <= rate_match.start() < end or start <= tenure_match.start() < end:
            continue
        principal = _amount_to_inr(match.group(1), match.group(2))
        break
    if principal is None:
        raise ValueError("Include a loan principal, such as ₹10 lakh.")
    return {
        "principal": principal,
        "interest_rate": float(rate_match.group(1)),
        "tenure_years": float(tenure_match.group(1)),
    }


def _parse_labeled_amount(text: str, label_pattern: str) -> float:
    match = re.search(
        r"(?:" + label_pattern + r")"
        + r"\s*(?:is|of|=|:)?\s*(?:₹|rs\.?\s*|rupees?\s*)?"
        + r"([\d,]+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|cr|k)?\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        raise ValueError
    return _amount_to_inr(match.group(1), match.group(2))


def parse_eligibility_question(question: str) -> dict[str, float]:
    try:
        salary = _parse_labeled_amount(question, r"salary|monthly income")
        existing_emi = _parse_labeled_amount(question, r"existing\s+emi|current\s+emi")
    except ValueError as exc:
        raise ValueError("Include both monthly salary and existing EMI amounts.") from exc
    return {"salary": salary, "existing_emi": existing_emi}


def build_banking_workflow(rag, vector_store):
    """Build a StateGraph that routes FAQs, EMI, and eligibility deterministically."""
    from langgraph.graph import END, START, StateGraph

    def classify(state: BankingState):
        question = (state.get("question") or "").strip()
        if not question:
            raise ValueError("Question cannot be empty")
        return {"intent": classify_intent(question)}

    def faq_node(state: BankingState):
        result = rag.answer(state["question"])
        documents = result["retrieved_docs"]
        return {
            "context": "\n\n".join(document.page_content for document in documents),
            "retrieved_docs": documents,
            "answer": result["answer"],
            "sources": result["sources"],
        }

    def emi_node(state: BankingState):
        details = parse_emi_question(state["question"])
        result = calculate_emi(**details)
        answer = (
            f"Estimated monthly EMI: ₹{result['monthly_emi']:,.2f}. "
            "This is a mathematical estimate, not a bank offer."
        )
        return {
            "context": "Standard reducing-balance EMI calculation.",
            "retrieved_docs": [],
            "answer": answer,
            "sources": ["Calculation tool (standard EMI formula; not a bank document)."],
        }

    def eligibility_node(state: BankingState):
        details = parse_eligibility_question(state["question"])
        result = check_eligibility(**details)
        decision = "meets" if result["eligible"] else "does not meet"
        answer = (
            f"Based on the sample rules, this income profile {decision} the stated "
            f"criteria. {result['reason']} Debt ratio: {result['debt_ratio']:.1%}. "
            "This is not a bank approval or credit decision."
        )
        return {
            "context": "Eligibility rules configured for this demonstration.",
            "retrieved_docs": [],
            "answer": answer,
            "sources": ["Eligibility tool (configured sample rules; not a bank document)."],
        }

    workflow = StateGraph(BankingState)
    workflow.add_node("classify", classify)
    workflow.add_node("faq", faq_node)
    workflow.add_node("emi", emi_node)
    workflow.add_node("eligibility", eligibility_node)
    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges(
        "classify",
        lambda state: state["intent"],
        {"faq": "faq", "emi": "emi", "eligibility": "eligibility"},
    )
    workflow.add_edge("faq", END)
    workflow.add_edge("emi", END)
    workflow.add_edge("eligibility", END)
    return workflow.compile()


def create_agent_executor(rag, vector_store, api_key: str | None = None):
    """Create a tool-calling AgentExecutor for applications needing LLM routing."""
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    from rag_chain import create_llm

    tools = [
        build_emi_tool(),
        build_eligibility_tool(),
        build_faq_tool(rag),
        build_document_search_tool(vector_store),
    ]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional banking assistant. Use the FAQ and document "
                "search tools for policy facts; never answer policy questions from "
                "memory or model knowledge. Use the calculator and eligibility "
                "tools for their respective requests. Clearly label calculations "
                "and sample eligibility checks as estimates, not bank decisions.",
            ),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(create_llm(api_key), tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=6,
        handle_parsing_errors=True,
        verbose=False,
    )