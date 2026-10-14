from types import SimpleNamespace

import pytest

from agent import (
    build_banking_workflow,
    classify_intent,
    parse_eligibility_question,
    parse_emi_question,
)


class FakeRAG:
    def answer(self, question):
        document = SimpleNamespace(
            page_content="Home-loan applicants need identity proof.",
            metadata={"source": "home_loan.pdf", "page": 0},
        )
        return {
            "answer": "Submit identity proof.",
            "retrieved_docs": [document],
            "sources": ["✅ home_loan.pdf – Page 1"],
        }


def test_classifies_requested_tool_paths():
    assert classify_intent("What documents for a home loan?") == "faq"
    assert classify_intent("Calculate EMI for 10 lakh") == "emi"
    assert classify_intent("Am I eligible for a loan?") == "eligibility"


def test_parses_sample_natural_language_inputs():
    assert parse_emi_question("Calculate EMI for ₹10 lakh loan for 5 years at 9%.") == {
        "principal": 1_000_000,
        "interest_rate": 9,
        "tenure_years": 5,
    }
    assert parse_eligibility_question(
        "Am I eligible if salary is ₹60,000 and existing EMI is ₹10,000?"
    ) == {"salary": 60_000, "existing_emi": 10_000}


def test_langgraph_routes_faq_and_emi_and_eligibility():
    workflow = build_banking_workflow(FakeRAG(), object())

    faq = workflow.invoke({"question": "What documents for a home loan?"})
    emi = workflow.invoke({"question": "Calculate EMI for ₹10 lakh for 5 years at 9%"})
    eligible = workflow.invoke(
        {"question": "Am I eligible if salary is 60000 and existing EMI is 10000?"}
    )

    assert faq["sources"] == ["✅ home_loan.pdf – Page 1"]
    assert "₹20,758.36" in emi["answer"]
    assert "meets" in eligible["answer"]


def test_workflow_rejects_empty_question():
    workflow = build_banking_workflow(FakeRAG(), object())
    with pytest.raises(ValueError, match="Question cannot be empty"):
        workflow.invoke({"question": " "})