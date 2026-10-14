from types import SimpleNamespace

import pytest

from retriever import format_sources, get_retriever, retrieve_documents


class FakeVectorStore:
    def __init__(self):
        self.calls = []

    def as_retriever(self, search_kwargs, search_type):
        return search_type, search_kwargs

    def similarity_search(self, question, k):
        self.calls.append((question, k))
        return [SimpleNamespace(page_content="answer", metadata={"source": "faq.pdf", "page": 1})]


def test_retriever_uses_requested_top_k():
    store = FakeVectorStore()
    assert get_retriever(store, k=3) == (
        "similarity_score_threshold",
        {"k": 3, "score_threshold": 0.15},
    )
    assert retrieve_documents(store, "account balance", k=3)[0].page_content == "answer"
    assert store.calls == [("account balance", 3)]


def test_rejects_empty_queries_and_invalid_k():
    store = FakeVectorStore()
    with pytest.raises(ValueError, match="Question cannot be empty"):
        retrieve_documents(store, " ")
    with pytest.raises(ValueError, match="at least 1"):
        get_retriever(store, k=0)


def test_formats_page_as_one_based_and_deduplicates_sources():
    source = SimpleNamespace(metadata={"source": "faq.pdf", "page": 1})
    assert format_sources([source, source]) == ["✅ faq.pdf – Page 2"]