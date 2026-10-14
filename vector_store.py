"""OpenAI embedding and persistent Chroma index helpers."""

from pathlib import Path

from document_loader import load_documents
from text_splitter import split_documents


class VectorStoreError(RuntimeError):
    """Raised when the vector index cannot be created or opened."""


def _embeddings(api_key: str | None = None):
    try:
        from langchain_openai import OpenAIEmbeddings

        kwargs = {"model": "text-embedding-3-small"}
        if api_key:
            kwargs["api_key"] = api_key
        return OpenAIEmbeddings(**kwargs)
    except Exception as exc:
        raise VectorStoreError(
            "Could not initialize OpenAI embeddings. Check OPENAI_API_KEY and "
            "the embedding dependency."
        ) from exc


def create_vector_store(
    documents=None,
    persist_directory: str | Path = "chroma_db",
    api_key: str | None = None,
):
    """Create or replace the persistent Chroma collection from documents."""
    try:
        from langchain_chroma import Chroma

        source_documents = documents if documents is not None else load_documents()
        chunks = split_documents(source_documents, chunk_size=500, chunk_overlap=50)
        embeddings = _embeddings(api_key)
        directory = Path(persist_directory)
        directory.mkdir(parents=True, exist_ok=True)

        existing = Chroma(
            collection_name="banking_documents",
            persist_directory=str(directory),
            embedding_function=embeddings,
        )
        existing.delete_collection()
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="banking_documents",
            persist_directory=str(directory),
        )
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(f"Could not create Chroma vector store: {exc}") from exc


def load_vector_store(
    persist_directory: str | Path = "chroma_db",
    api_key: str | None = None,
):
    """Open an existing persistent collection, failing clearly if it is empty."""
    directory = Path(persist_directory)
    if not directory.is_dir():
        raise VectorStoreError(f"Chroma database does not exist: {directory}")
    try:
        from langchain_chroma import Chroma

        store = Chroma(
            collection_name="banking_documents",
            persist_directory=str(directory),
            embedding_function=_embeddings(api_key),
        )
        if store._collection.count() == 0:
            raise VectorStoreError("Chroma database contains no indexed documents")
        return store
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(f"Could not load Chroma vector store: {exc}") from exc