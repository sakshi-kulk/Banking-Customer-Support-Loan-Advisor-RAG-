from types import SimpleNamespace

from tools.document_search_tool import search_policy_documents


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents

    def invoke(self, query):
        return self.documents


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.kwargs = None

    def as_retriever(self, search_type, search_kwargs):
        self.kwargs = (search_type, search_kwargs)
        return FakeRetriever(self.documents)


def test_search_keeps_each_passage_and_source_with_threshold():
    documents = [
        SimpleNamespace(page_content="First passage", metadata={"source": "policy.pdf", "page": 0}),
        SimpleNamespace(page_content="Second passage", metadata={"source": "policy.pdf", "page": 0}),
    ]
    store = FakeVectorStore(documents)

    result = search_policy_documents(store, "policy question", "policy.pdf")

    assert result.count("policy.pdf – Page 1") == 2
    assert "First passage" in result and "Second passage" in result
    assert store.kwargs == (
        "similarity_score_threshold",
        {"k": 3, "score_threshold": 0.15, "filter": {"source": "policy.pdf"}},
    )