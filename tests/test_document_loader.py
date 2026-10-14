from types import SimpleNamespace

import pytest

from document_loader import DocumentLoadError, load_documents


def test_loads_markdown_sample_with_citation_metadata(tmp_path):
    (tmp_path / "banking_faq.md").write_text("Sample account FAQ.", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].page_content == "Sample account FAQ."
    assert documents[0].metadata == {"source": "banking_faq.md", "page": 0}


def test_uses_pdf_loader_and_normalizes_source(monkeypatch, tmp_path):
    pdf = tmp_path / "policy.pdf"
    pdf.touch()
    page = SimpleNamespace(page_content="Policy page", metadata={"page": 2})

    class FakePDFLoader:
        def __init__(self, path):
            assert path == str(pdf)

        def load(self):
            return [page]

    import langchain_community.document_loaders

    monkeypatch.setattr(
        langchain_community.document_loaders, "PyPDFLoader", FakePDFLoader
    )

    documents = load_documents(tmp_path)

    assert documents[0].metadata == {"page": 2, "source": "policy.pdf"}


def test_raises_actionable_error_when_folder_has_no_documents(tmp_path):
    with pytest.raises(DocumentLoadError, match="No PDF, Markdown, or text"):
        load_documents(tmp_path)