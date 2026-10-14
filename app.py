"""Streamlit UI for the AI Banking Customer Support & Loan Advisor."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import build_banking_workflow
from document_loader import load_documents
from rag_chain import BankingRAG
from vector_store import create_vector_store, load_vector_store

ROOT = Path(__file__).resolve().parent
DOCUMENT_DIR = ROOT / "data" / "documents"
VECTOR_DIR = ROOT / "chroma_db"
load_dotenv(ROOT / ".env")


def render_sources(sources):
    if sources:
        st.markdown("**Sources**")
        for source in sources:
            st.markdown(f"- {source}")


st.set_page_config(page_title="AI Banking Assistant", page_icon="🏦", layout="centered")
st.title("🏦 AI Banking Assistant")
st.caption("Customer support and loan guidance grounded in your indexed banking documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "workflow" not in st.session_state:
    st.session_state.workflow = None
if "index_error" not in st.session_state:
    st.session_state.index_error = ""

with st.sidebar:
    st.subheader("Document index")
    api_key = st.text_input(
        "OpenAI API key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Used for embeddings and GPT-4o-mini. The key is not stored in the index.",
    )
    uploads = st.file_uploader(
        "Upload banking PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if st.button("Build / refresh index", type="primary", use_container_width=True):
        if not api_key:
            st.error("Add an OpenAI API key to create embeddings and answer questions.")
        else:
            try:
                for uploaded_file in uploads or []:
                    safe_name = Path(uploaded_file.name).name
                    (DOCUMENT_DIR / safe_name).write_bytes(uploaded_file.getvalue())
                with st.spinner("Loading documents and building the Chroma index..."):
                    documents = load_documents(DOCUMENT_DIR)
                    vector_store = create_vector_store(
                        documents,
                        persist_directory=VECTOR_DIR,
                        api_key=api_key,
                    )
                    rag = BankingRAG(vector_store, api_key=api_key)
                    st.session_state.vector_store = vector_store
                    st.session_state.rag = rag
                    st.session_state.workflow = build_banking_workflow(rag, vector_store)
                    st.session_state.index_error = ""
                st.success(f"Indexed {len(documents)} document page(s).")
            except Exception as exc:
                st.session_state.index_error = str(exc)
                st.error(f"Indexing failed: {exc}")

    if st.session_state.index_error and st.session_state.workflow is None:
        st.caption(st.session_state.index_error)

    st.divider()
    st.caption(
        "Bundled Markdown files are fictional training samples, not official bank policies. "
        "Replace them with approved documents before customer use."
    )

if st.session_state.workflow is None and api_key and VECTOR_DIR.is_dir():
    try:
        vector_store = load_vector_store(VECTOR_DIR, api_key=api_key)
        rag = BankingRAG(vector_store, api_key=api_key)
        st.session_state.vector_store = vector_store
        st.session_state.rag = rag
        st.session_state.workflow = build_banking_workflow(rag, vector_store)
    except Exception as exc:
        st.session_state.index_error = f"Saved index could not be loaded: {exc}"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

with st.form("question_form", clear_on_submit=True):
    question = st.text_input(
        "Ask a banking question",
        placeholder="e.g. What documents are required for a home loan?",
    )
    submitted = st.form_submit_button("Ask AI", type="primary", use_container_width=True)

if submitted:
    if not question.strip():
        st.warning("Enter a question before submitting.")
    elif st.session_state.workflow is None:
        st.warning("Build the document index first. An OpenAI API key is required.")
    else:
        st.session_state.messages.append({"role": "user", "content": question.strip()})
        with st.spinner("Checking the relevant banking documents..."):
            try:
                result = st.session_state.workflow.invoke(
                    {
                        "question": question.strip(),
                        "chat_history": st.session_state.messages[:-1],
                    }
                )
                response = result["answer"]
                sources = result.get("sources", [])
            except Exception as exc:
                response = f"I couldn't process that request: {exc}"
                sources = []
        st.session_state.messages.append(
            {"role": "assistant", "content": response, "sources": sources}
        )
        st.rerun()