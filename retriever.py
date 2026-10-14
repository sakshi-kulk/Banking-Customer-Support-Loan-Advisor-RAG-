"""Small helpers for retrieving source passages from Chroma."""


def get_retriever(vector_store, k: int = 3):
    if k < 1:
        raise ValueError("k must be at least 1")
    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": 0.15},
    )


def retrieve_documents(vector_store, question: str, k: int = 3):
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty")
    return vector_store.similarity_search(question, k=k)


def format_sources(documents):
    sources = []
    seen = set()
    for document in documents:
        metadata = document.metadata or {}
        filename = metadata.get("source", "Unknown document")
        page = metadata.get("page")
        page_label = f"Page {int(page) + 1}" if page is not None else "Page unknown"
        citation = f"{filename} – {page_label}"
        if citation not in seen:
            seen.add(citation)
            sources.append(f"✅ {citation}")
    return sources