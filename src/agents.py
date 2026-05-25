from __future__ import annotations

from typing import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from .rag_engine import RAGEngine


def _web_search(query: str) -> str:
    """Search the web via DuckDuckGo (no API key required)."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No web results found."
        return "\n\n".join(f"**{r['title']}**\n{r['body']}" for r in results)
    except Exception as exc:
        return f"Web search unavailable: {exc}"


class ResearchState(TypedDict):
    query: str
    use_docs: bool
    use_web: bool
    doc_results: str
    web_results: str
    final_report: str


_SYNTHESIS_PROMPT = """You are a senior research analyst. Based on the information below, write a
well-structured research report in Markdown with these sections:

1. Executive Summary
2. Key Findings
3. Analysis & Insights
4. Conclusions & Recommendations

Research Topic: {query}

Information Gathered:
{context}"""


class ResearchAgent:
    """
    Multi-agent research pipeline built with LangGraph.

    Graph: document_research → web_research → synthesize → END
    """

    def __init__(self, api_key: str, rag_engine: RAGEngine | None = None):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3,
        )
        self.rag_engine = rag_engine
        self._graph = self._build_graph()

    # ── LangGraph nodes ──────────────────────────────────────────────────────

    def _doc_research_node(self, state: ResearchState) -> ResearchState:
        if state.get("use_docs") and self.rag_engine:
            answer, _ = self.rag_engine.query(state["query"])
            state["doc_results"] = answer
        else:
            state["doc_results"] = ""
        return state

    def _web_research_node(self, state: ResearchState) -> ResearchState:
        if state.get("use_web"):
            state["web_results"] = _web_search(state["query"])
        else:
            state["web_results"] = ""
        return state

    def _synthesis_node(self, state: ResearchState) -> ResearchState:
        parts: list[str] = []
        if state.get("doc_results"):
            parts.append(f"**From Uploaded Documents:**\n{state['doc_results']}")
        if state.get("web_results"):
            parts.append(f"**From Web Search:**\n{state['web_results']}")

        context = "\n\n---\n\n".join(parts) if parts else "No information available."
        prompt = _SYNTHESIS_PROMPT.format(query=state["query"], context=context)

        response = self.llm.invoke([("human", prompt)])
        state["final_report"] = response.content
        return state

    def _build_graph(self):
        workflow = StateGraph(ResearchState)
        workflow.add_node("document_research", self._doc_research_node)
        workflow.add_node("web_research", self._web_research_node)
        workflow.add_node("synthesize", self._synthesis_node)

        workflow.set_entry_point("document_research")
        workflow.add_edge("document_research", "web_research")
        workflow.add_edge("web_research", "synthesize")
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    def research(self, query: str, use_docs: bool = True, use_web: bool = True) -> str:
        initial: ResearchState = {
            "query": query,
            "use_docs": use_docs,
            "use_web": use_web,
            "doc_results": "",
            "web_results": "",
            "final_report": "",
        }
        final = self._graph.invoke(initial)
        return final["final_report"]
