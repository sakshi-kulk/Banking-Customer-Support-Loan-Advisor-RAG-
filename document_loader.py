"""Load banking source documents while preserving citation metadata."""

from pathlib import Path


class DocumentLoadError(RuntimeError):
    """Raised when no usable banking documents can be loaded."""


def load_documents(document_dir: str | Path = "data/documents"):
    """Load every PDF, Markdown, and text document in ``document_dir``.

    PDF pages are loaded with PyPDFLoader. Markdown/text files are supported
    for the bundled sample material and are marked as page 1 for citations.
    """
    directory = Path(document_dir)
    if not directory.is_dir():
        raise DocumentLoadError(f"Document folder does not exist: {directory}")

    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".md", ".txt"}
    )
    if not paths:
        raise DocumentLoadError(
            f"No PDF, Markdown, or text documents found in {directory}. "
            "Add bank documents before asking policy questions."
        )

    documents = []
    failures = []
    for path in paths:
        try:
            if path.suffix.lower() == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader

                loaded = PyPDFLoader(str(path)).load()
                for document in loaded:
                    document.metadata["source"] = path.name
                documents.extend(loaded)
            else:
                from langchain_core.documents import Document

                content = path.read_text(encoding="utf-8").strip()
                if content:
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={"source": path.name, "page": 0},
                        )
                    )
        except Exception as exc:
            failures.append(f"{path.name}: {exc}")

    if not documents:
        detail = "; ".join(failures) if failures else "all supported files were empty"
        raise DocumentLoadError(f"Unable to load documents from {directory}: {detail}")
    return documents