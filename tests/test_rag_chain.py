from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
import math

from rag_chain import BankingRAG, create_retrieval_qa


class TinyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        vector = [0.0] * 8
        for index, value in enumerate(text.encode("utf-8")):
            vector[index % len(vector)] += value / 255
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]


def make_store(tmp_path):
    return Chroma.from_documents(
        documents=[
            Document(
                page_content="The sample savings account requires a minimum average monthly balance of INR 5,000.",
                metadata={"source": "banking_faq.pdf", "page": 0},
            )
        ],
        embedding=TinyEmbeddings(),
        collection_name="banking_test",
        persist_directory=str(tmp_path),
    )


def test_retrieval_qa_returns_answer_and_source_documents(tmp_path):
    chain = create_retrieval_qa(
        FakeListChatModel(responses=["The sample minimum is INR 5,000."]),
        make_store(tmp_path),
    )

    result = chain.invoke({"query": "minimum savings balance"})

    assert result["result"] == "The sample minimum is INR 5,000."
    assert result["source_documents"][0].metadata["source"] == "banking_faq.pdf"


def test_conversational_chain_retains_follow_up_memory_and_citations(tmp_path):
    assistant = BankingRAG(
        make_store(tmp_path),
        llm=FakeListChatModel(
            responses=[
                "The sample minimum is INR 5,000.",
                "What is the tenure for a savings account?",
                "The documents do not specify a savings account tenure.",
            ]
        ),
    )

    first = assistant.answer("What is the minimum savings balance?")
    follow_up = assistant.answer("What is its tenure?")

    assert first["answer"] == "The sample minimum is INR 5,000."
    assert follow_up["answer"] == "The documents do not specify a savings account tenure."
    assert follow_up["sources"] == ["✅ banking_faq.pdf – Page 1"]
    assert "minimum savings balance" in str(assistant.chain.memory.buffer).lower()