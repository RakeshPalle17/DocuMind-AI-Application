from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


_QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a knowledgeable AI assistant. Answer the question using ONLY the provided context.
If the answer is not found in the context, clearly state that.

Context:
{context}""",
    ),
    ("human", "{input}"),
])

_SUMMARY_STYLES: dict[str, str] = {
    "Executive Summary": "Write a concise executive summary in 3–5 paragraphs covering the most important points.",
    "Bullet Points": "Summarize the content as a structured bullet-point list organized by theme.",
    "Detailed Analysis": "Provide a comprehensive analysis: themes, key findings, evidence, and implications.",
    "Key Takeaways": "Extract the 5–10 most critical takeaways, ordered by importance.",
}


class RAGEngine:
    """Manages document embeddings, vector storage, and retrieval-augmented generation."""

    def __init__(self, api_key: str):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key,
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.2,
        )
        self.vectorstore: Chroma | None = None

    def add_documents(self, documents: list[Document]) -> None:
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )

    def query(self, question: str) -> tuple[str, list[dict]]:
        if not self.vectorstore:
            return "No documents loaded. Please upload documents first.", []

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )
        chain = create_retrieval_chain(
            retriever,
            create_stuff_documents_chain(self.llm, _QA_PROMPT),
        )

        result = chain.invoke({"input": question})
        answer: str = result["answer"]
        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "content": doc.page_content,
            }
            for doc in result.get("context", [])
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
        messages = [
            ("system", f"You are an expert analyst. {instruction} Format your response in Markdown."),
            ("human", f"Analyze and summarize the following content:\n\n{text}"),
        ]
        return self.llm.invoke(messages).content
