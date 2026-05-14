#!/usr/bin/env python3
"""
Streamlit Dashboard for RAG Knowledge Base
Interactive interface: input question → show retrieved docs → generated answer
"""

import streamlit as st
import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline

# Page config
st.set_page_config(
    page_title="ArXiv RAG Knowledge Base",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .source-card {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    .source-title {
        font-weight: 600;
        color: #111827;
        font-size: 0.95rem;
    }
    .source-meta {
        color: #6b7280;
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }
    .relevance-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .answer-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stat-card {
        background-color: #f3f4f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #6b7280;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_pipeline():
    """Initialize RAG pipeline (cached)."""
    return RAGPipeline()


def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Pipeline Settings")
        use_reranker = st.toggle("Cross-encoder Re-ranking", value=True)
        use_hyde = st.toggle("HyDE (Hypothetical Document Embedding)", value=True)
        use_llm_transform = st.toggle("LLM Query Transformation", value=True)
        
        st.markdown("---")
        st.markdown("### 📊 Pipeline Architecture")
        st.markdown("""
        1. **Query Transformation** — Expand & decompose queries
        2. **HyDE** — Generate hypothetical document
        3. **First-stage Retrieval** — FAISS similarity search
        4. **Cross-encoder Re-ranking** — Precision scoring
        5. **Answer Synthesis** — Cited generation
        """)
        
        st.markdown("---")
        st.markdown("### 📚 Corpus")
        st.markdown("2,794 real ArXiv abstracts  
ML, NLP, CV, Data Science, RL")
    
    # Main content
    st.markdown('<div class="main-header">📚 ArXiv RAG Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Production-grade RAG system for scientific literature search. Ask anything about machine learning, NLP, computer vision, or data science.</div>', unsafe_allow_html=True)
    
    # Query input
    query = st.text_input(
        "Ask a research question...",
        placeholder="e.g., What are the latest techniques in transformer architecture?",
        key="query_input"
    )
    
    # Example queries
    cols = st.columns(3)
    examples = [
        "What is attention mechanism in transformers?",
        "How does few-shot learning work?",
        "What are diffusion models?"
    ]
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex_{i}"):
            st.session_state.query_input = ex
            query = ex
    
    if query:
        # Initialize pipeline with selected settings
        try:
            pipeline = RAGPipeline(
                use_reranker=use_reranker,
                use_hyde=use_hyde,
                use_llm_transform=use_llm_transform
            )
            
            with st.spinner("🔍 Searching ArXiv corpus..."):
                result = pipeline.query(query)
            
            # Stats row
            stat_cols = st.columns(4)
            stats = [
                (f"{result.num_retrieved}", "Candidates Retrieved"),
                (f"{result.num_reranked}", "Sources Used"),
                (f"{result.retrieval_time_ms:.0f}ms", "Retrieval Time"),
                (f"{result.total_time_ms:.0f}ms", "Total Time"),
            ]
            for col, (val, label) in zip(stat_cols, stats):
                col.markdown(f'<div class="stat-card"><div class="stat-value">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)
            
            # Answer
            st.markdown("### 📝 Answer")
            st.markdown(f'<div class="answer-box">{result.answer}</div>', unsafe_allow_html=True)
            
            # Sources
            st.markdown("### 📄 Sources")
            for i, src in enumerate(result.sources, 1):
                score = src['relevance_score']
                score_color = "#10b981" if score > 0.8 else "#f59e0b" if score > 0.5 else "#ef4444"
                
                authors = ", ".join(src['authors'][:2]) if src['authors'] else "Unknown"
                year = src['published'][:4] if src['published'] else ""
                
                st.markdown(f"""
                <div class="source-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <span class="source-title">[{i}] {src['title']}</span>
                        <span class="relevance-badge" style="background-color: {score_color}20; color: {score_color};">Relevance: {score}</span>
                    </div>
                    <div class="source-meta">{authors} {f'({year})' if year else ''}</div>
                    <div style="margin-top: 0.5rem; color: #4b5563; font-size: 0.9rem;">{src['content']}</div>
                    <div style="margin-top: 0.5rem;"><a href="{src['url']}" target="_blank">View on ArXiv →</a></div>
                </div>
                """, unsafe_allow_html=True)
            
            # Strategy used
            st.markdown(f"<div style='color: #9ca3af; font-size: 0.8rem; margin-top: 1rem;'>Pipeline: {result.strategy_used}</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure the vector store is built. Run: `python src/embeddings.py`")

    # Footer
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #9ca3af; font-size: 0.8rem;'>Built with real ArXiv data • 2,794 abstracts • FAISS + SentenceTransformers</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
