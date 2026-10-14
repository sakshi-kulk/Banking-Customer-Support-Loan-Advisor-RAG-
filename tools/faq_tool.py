"""RAG-backed FAQ search tool factory."""


def build_faq_tool(rag):
    from langchain_core.tools import StructuredTool

    def retrieve_banking_faq(question: str) -> str:
        result = rag.answer(question)
        sources = "\n".join(result["sources"]) or "No document sources found."
        return f"{result['answer']}\n\nSources:\n{sources}"

    return StructuredTool.from_function(
        func=retrieve_banking_faq,
        name="faq_retrieval",
        description="Answer a banking policy or product FAQ using only retrieved bank documents and return citations.",
    )