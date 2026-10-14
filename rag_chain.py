"""Grounded single-turn and conversational question-answering chains."""

from retriever import format_sources, get_retriever

FALLBACK_ANSWER = "I don't have enough information in the bank documents to answer this question."
BANKING_PROMPT = """You are a professional banking assistant.
Answer customer questions using only the provided context. Do not use outside
knowledge or infer bank policy. If the answer is not supported by the context,
respond exactly: "I don't have enough information in the bank documents to answer this question."
Keep guidance clear and concise. Do not treat sample training documents as real
bank offers; explain that their terms are illustrative when relevant.

Context:
{context}

Question: {question}
Answer:"""


class RAGError(RuntimeError):
    """Raised when the model or retrieval chain fails."""


def create_llm(api_key: str | None = None):
    try:
        from langchain_openai import ChatOpenAI

        kwargs = {"model": "gpt-4o-mini", "temperature": 0}
        if api_key:
            kwargs["api_key"] = api_key
        return ChatOpenAI(**kwargs)
    except Exception as exc:
        raise RAGError(f"Could not initialize OpenAI chat model: {exc}") from exc


def _qa_prompt():
    from langchain_core.prompts import PromptTemplate

    return PromptTemplate(
        template=BANKING_PROMPT,
        input_variables=["context", "question"],
    )


def create_retrieval_qa(llm, vector_store, k: int = 3):
    """Build the requested RetrievalQA single-question chain."""
    from langchain.chains import RetrievalQA

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=get_retriever(vector_store, k=k),
        return_source_documents=True,
        chain_type_kwargs={"prompt": _qa_prompt()},
    )


def create_conversational_retrieval_chain(llm, vector_store, memory=None, k: int = 3):
    """Build ConversationalRetrievalChain with persistent in-process memory."""
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory

    if memory is None:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True,
        )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=get_retriever(vector_store, k=k),
        memory=memory,
        return_source_documents=True,
        output_key="answer",
        response_if_no_docs_found=FALLBACK_ANSWER,
        combine_docs_chain_kwargs={"prompt": _qa_prompt()},
    )


class BankingRAG:
    def __init__(self, vector_store, llm=None, memory=None, k: int = 3, api_key=None):
        self.vector_store = vector_store
        self.llm = llm or create_llm(api_key)
        self.memory = memory
        self.chain = create_conversational_retrieval_chain(
            self.llm, vector_store, memory=memory, k=k
        )

    def answer(self, question: str):
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        try:
            result = self.chain.invoke({"question": question.strip()})
            documents = result.get("source_documents", [])
            answer = (result.get("answer") or "").strip()
            if not documents or not answer:
                answer = FALLBACK_ANSWER
            return {
                "answer": answer,
                "retrieved_docs": documents,
                "sources": format_sources(documents),
            }
        except ValueError:
            raise
        except Exception as exc:
            raise RAGError(f"Banking question could not be answered: {exc}") from exc