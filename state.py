"""Shared state shape for Streamlit conversations and LangGraph."""

from typing import Any, TypedDict


class BankingState(TypedDict, total=False):
    question: str
    intent: str
    context: str
    retrieved_docs: list[Any]
    answer: str
    sources: list[str]
    chat_history: list[dict[str, str]]