"""Semantic search over banking policy documents."""

from retriever import format_sources


def search_policy_documents(vector_store, query: str, document_name: str = "", k: int = 3):
    if not query.strip():
        raise ValueError("Search query cannot be empty")
    search_kwargs = {"k": k, "score_threshold": 0.15}
    if document_name.strip():
        search_kwargs["filter"] = {"source": document_name.strip()}
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs=search_kwargs,
    )
    documents = retriever.invoke(query)
    if not documents:
        return "No matching passages were found in the bank documents."
    passages = [
        f"{format_sources([document])[0]}\n{document.page_content}"
        for document in documents
    ]
    return "\n\n".join(passages)


def build_document_search_tool(vector_store):
    from langchain_core.tools import StructuredTool

    def search_banking_documents(query: str, document_name: str = "") -> str:
        return search_policy_documents(vector_store, query, document_name)

    return StructuredTool.from_function(
        func=search_banking_documents,
        name="document_search",
        description="Search bank documents for policy passages, optionally restricting the search to an exact source filename.",
    )