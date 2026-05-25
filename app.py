from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Page config must be the very first Streamlit call
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "**DocuMind AI** — RAG-powered Document Intelligence & Research Platform"},
)

sys.path.insert(0, str(Path(__file__).parent))

from src.document_processor import DocumentProcessor
from src.rag_engine import RAGEngine
from src.agents import ResearchAgent

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.hero-title {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; line-height: 1.2;
}
.hero-sub {
    text-align: center; color: #6b7280; font-size: 1.05rem; margin-bottom: 1.8rem;
}
.badge {
    display: inline-block; background: #667eea22; color: #667eea;
    border-radius: 20px; padding: 2px 10px; font-size: 0.78rem;
    font-weight: 600; margin: 2px;
}
.source-box {
    background: #f8fafc; border-left: 3px solid #667eea;
    padding: 0.6rem 1rem; border-radius: 0 6px 6px 0;
    margin: 4px 0; font-size: 0.85rem;
}
.feature-card {
    background: #f8fafc; border-radius: 12px; padding: 1.2rem 1rem;
    border: 1px solid #e2e8f0; text-align: center;
}
.stButton > button { border-radius: 8px; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state ─────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults: dict = {
        "messages": [],
        "rag_engine": None,
        "agent": None,
        "doc_names": [],
        "docs_ready": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _get_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GOOGLE_API_KEY", "")  # type: ignore[attr-defined]
        except Exception:
            pass
    return key


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🧠 DocuMind AI")
        st.markdown(
            '<span class="badge">RAG</span>'
            '<span class="badge">LangChain</span>'
            '<span class="badge">LangGraph</span>'
            '<span class="badge">Gemini</span>'
            '<span class="badge">ChromaDB</span>',
            unsafe_allow_html=True,
        )
        st.divider()

        # API Key section
        st.markdown("### ⚙️ API Key")
        api_key = _get_api_key()
        if not api_key:
            api_key = st.text_input(
                "Google Gemini API Key",
                type="password",
                placeholder="AIza...",
                help="Free — 1,500 requests/day at Google AI Studio",
            )
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
            else:
                st.caption("👉 [Get free API key →](https://aistudio.google.com/app/apikey)")
        else:
            st.success("✅ API key loaded", icon="🔑")

        st.divider()

        # Document upload
        st.markdown("### 📄 Upload Documents")
        uploads = st.file_uploader(
            "Upload PDFs or text files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploads and api_key:
            if st.button("⚡ Process Documents", type="primary", use_container_width=True):
                _process_uploads(uploads, api_key)
        elif uploads and not api_key:
            st.caption("Enter API key above to process documents.")

        if st.session_state.docs_ready:
            st.success(f"✅ {len(st.session_state.doc_names)} document(s) ready")
            for name in st.session_state.doc_names:
                st.caption(f"📄 {name}")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.messages:
                st.metric("Messages", len(st.session_state.messages))
        with col2:
            if st.session_state.docs_ready:
                st.metric("Docs", len(st.session_state.doc_names))

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return api_key


def _process_uploads(uploads, api_key: str) -> None:
    processor = DocumentProcessor()
    all_chunks = []
    bar = st.sidebar.progress(0, text="Loading…")

    try:
        for i, f in enumerate(uploads):
            suffix = "." + f.name.rsplit(".", 1)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

            chunks = processor.process_file(tmp_path, f.name)
            all_chunks.extend(chunks)
            os.unlink(tmp_path)
            bar.progress((i + 1) / len(uploads), text=f"Processed: {f.name}")

        bar.progress(1.0, text="Building vector index…")
        rag = RAGEngine(api_key=api_key)
        rag.add_documents(all_chunks)

        st.session_state.rag_engine = rag
        st.session_state.agent = ResearchAgent(api_key=api_key, rag_engine=rag)
        st.session_state.doc_names = [f.name for f in uploads]
        st.session_state.docs_ready = True
        bar.empty()
        st.rerun()

    except Exception as exc:
        bar.empty()
        st.sidebar.error(f"❌ {exc}")


# ── Tab 1: Document Chat ──────────────────────────────────────────────────────
def tab_chat() -> None:
    if not st.session_state.docs_ready:
        st.info("👈 Upload documents in the sidebar to start chatting.")
        _render_feature_overview()
        return

    st.caption(f"Chatting with: **{', '.join(st.session_state.doc_names)}**")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    for s in msg["sources"]:
                        page = s.get("page", "N/A")
                        st.markdown(
                            f'<div class="source-box">'
                            f'<strong>{s["source"]}</strong> · page {page}<br>'
                            f'<small>{s["content"][:250]}…</small></div>',
                            unsafe_allow_html=True,
                        )

    if prompt := st.chat_input("Ask anything about your documents…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching & reasoning…"):
                answer, sources = st.session_state.rag_engine.query(prompt)
            st.markdown(answer)
            if sources:
                with st.expander("📚 Sources", expanded=False):
                    for s in sources:
                        page = s.get("page", "N/A")
                        st.markdown(
                            f'<div class="source-box">'
                            f'<strong>{s["source"]}</strong> · page {page}<br>'
                            f'<small>{s["content"][:250]}…</small></div>',
                            unsafe_allow_html=True,
                        )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


def _render_feature_overview() -> None:
    st.markdown("### How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="feature-card">'
            "<h4>📚 RAG Q&amp;A</h4>"
            "<p>Upload any PDF and ask precise questions. Answers include cited page references.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="feature-card">'
            "<h4>🔬 Research Agents</h4>"
            "<p>LangGraph pipeline: Doc Search → Web Search → Synthesis. Three agents, one report.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="feature-card">'
            "<h4>📝 Smart Summarize</h4>"
            "<p>Executive summary, bullet points, deep-dive analysis — pick your format.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Tech stack:** `LangChain` · `LangGraph` · `ChromaDB` · `Google Gemini` · `Streamlit`")


# ── Tab 2: Research Agents ────────────────────────────────────────────────────
def tab_research(api_key: str) -> None:
    st.markdown("### 🔬 Multi-Agent Research Pipeline")
    st.markdown(
        "Powered by **LangGraph** — three specialized agents collaborate:\n"
        "> `Document Researcher` → `Web Researcher` → `Synthesis Analyst`"
    )

    if not api_key:
        st.warning("Please enter your Gemini API key in the sidebar.")
        return

    query = st.text_input(
        "Research topic or question",
        placeholder="e.g. 'What are the risk factors discussed in the uploaded report?'",
    )

    col1, col2 = st.columns(2)
    with col1:
        use_docs = st.checkbox(
            "Search uploaded documents",
            value=st.session_state.docs_ready,
            disabled=not st.session_state.docs_ready,
            help="Requires documents to be uploaded and processed",
        )
    with col2:
        use_web = st.checkbox(
            "Search the web (DuckDuckGo)",
            value=True,
            help="No API key required — uses DuckDuckGo",
        )

    if st.button("🚀 Run Research Agents", type="primary", disabled=not query):
        if not use_docs and not use_web:
            st.error("Enable at least one data source.")
            return

        with st.spinner("Agents running… document search → web search → synthesis"):
            try:
                if st.session_state.agent:
                    report = st.session_state.agent.research(
                        query, use_docs=use_docs, use_web=use_web
                    )
                else:
                    tmp_agent = ResearchAgent(api_key=api_key)
                    report = tmp_agent.research(query, use_docs=False, use_web=use_web)
            except Exception as exc:
                st.error(f"Agent error: {exc}")
                return

        st.markdown("---")
        st.markdown(report)
        slug = query[:30].replace(" ", "_")
        st.download_button(
            "📥 Download Report (.md)",
            data=report,
            file_name=f"research_{slug}.md",
            mime="text/markdown",
        )


# ── Tab 3: Summarize ──────────────────────────────────────────────────────────
def tab_summarize() -> None:
    st.markdown("### 📝 Document Summarizer")

    if not st.session_state.docs_ready:
        st.info("Upload and process documents in the sidebar to enable summarization.")
        return

    mode = st.selectbox(
        "Summary format",
        ["Executive Summary", "Bullet Points", "Detailed Analysis", "Key Takeaways"],
    )

    if st.button("✨ Generate Summary", type="primary"):
        with st.spinner(f"Generating {mode}…"):
            try:
                summary = st.session_state.rag_engine.summarize(mode)
            except Exception as exc:
                st.error(f"Summarization error: {exc}")
                return

        st.markdown(summary)
        st.download_button(
            "📥 Download (.md)",
            data=summary,
            file_name=f"summary_{mode.lower().replace(' ', '_')}.md",
            mime="text/markdown",
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    _init_state()

    st.markdown('<div class="hero-title">🧠 DocuMind AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">'
        "RAG-powered Document Intelligence &amp; Multi-Agent Research Platform"
        "</p>",
        unsafe_allow_html=True,
    )

    api_key = render_sidebar()

    t1, t2, t3 = st.tabs(["💬 Document Chat", "🔬 Research Agents", "📝 Summarize"])
    with t1:
        tab_chat()
    with t2:
        tab_research(api_key)
    with t3:
        tab_summarize()


if __name__ == "__main__":
    main()
