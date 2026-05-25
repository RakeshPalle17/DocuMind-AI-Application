from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

_QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. Answer the question using ONLY the provided context. "
        "If the answer is not in the context, clearly say so.\n\nContext:\n{context}",
    ),
    ("human", "{question}"),
])

_SUMMARY_STYLES: dict[str, str] = {
    "Executive Summary": "Write a concise executive summary in 3–5 paragraphs covering the most important points.",
    "Bullet Points": "Summarize the content as a structured bullet-point list organized by theme.",
    "Detailed Analysis": "Provide a comprehensive analysis: themes, key findings, evidence, and implications.",
    "Key Takeaways": "Extract the 5–10 most critical takeaways, ordered by importance.",
}


def _make_llm_and_embeddings(provider: str, api_key: str, model: str):
    """Return (llm, embeddings) for the given provider.

    Embeddings always use FastEmbed (local, no API quota, ~40MB one-time download).
    """
    from langchain_community.embeddings import FastEmbedEmbeddings
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    if provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=model, groq_api_key=api_key, temperature=0.2)
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.2)

    return llm, embeddings


class RAGEngine:
    """RAG pipeline: vector store retrieval + LLM answer generation."""

    def __init__(self, api_key: str, provider: str = "gemini", model: str = "gemini-2.0-flash"):
        self.llm, self.embeddings = _make_llm_and_embeddings(provider, api_key, model)
        self._chain = _QA_PROMPT | self.llm | StrOutputParser()
        self.vectorstore = None

    def add_documents(self, documents: list[Document]) -> None:
        from langchain_chroma import Chroma
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )

    def query(self, question: str) -> tuple[str, list[dict]]:
        if not self.vectorstore:
            return "No documents loaded. Please upload documents first.", []

        docs: list[Document] = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ).invoke(question)

        context = "\n\n".join(
            f"[Source: {d.metadata.get('source', 'Unknown')} | Page: {d.metadata.get('page', 'N/A')}]\n{d.page_content}"
            for d in docs
        )

        try:
            answer = self._chain.invoke({"context": context, "question": question})
        except Exception as exc:
            return _quota_or_error(exc), []

        sources = [
            {
                "source": d.metadata.get("source", "Unknown"),
                "page": d.metadata.get("page", "N/A"),
                "content": d.page_content,
            }
            for d in docs
        ]
        return answer, sources

    def get_document_text(self, max_chars: int = 12_000) -> str:
        if not self.vectorstore:
            return ""
        data = self.vectorstore.get()
        return "\n\n".join(data.get("documents", []))[:max_chars]

    def summarize(self, summary_type: str = "Executive Summary") -> str:
        text = self.get_document_text()
        if not text:
            return "No documents loaded."

        instruction = _SUMMARY_STYLES.get(summary_type, _SUMMARY_STYLES["Executive Summary"])
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are an expert analyst. {instruction} Format in Markdown."),
            ("human", "Content to analyze:\n\n{text}"),
        ])
        chain = prompt | self.llm | StrOutputParser()
        try:
            return chain.invoke({"text": text})
        except Exception as exc:
            return _quota_or_error(exc)


def _quota_or_error(exc: Exception) -> str:
    msg = str(exc)
    if "429" in msg or "quota" in msg.lower() or "exhausted" in msg.lower() or "resource_exhausted" in msg.lower():
        return (
            "⚠️ **API quota exceeded.** Switch to **Groq** in the sidebar "
            "(free, 14,400 req/day — get key at console.groq.com). "
            "Or wait a few minutes for rate limits to reset."
        )
    return f"❌ Error: {exc}"
