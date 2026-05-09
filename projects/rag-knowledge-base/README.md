## Project 2: RAG Knowledge Base

**Context:** Retrieval-Augmented Generation system for enterprise knowledge management inspired by internal document processing at Akin Gump and DC Government.

**Dataset:**
- [Federal regulations (eCFR)](https://www.ecfr.gov/)
- [DC Municipal Regulations](https://dcregs.dc.gov/)
- Simulated legal/policy document corpus

**Objective:** Build a RAG system that enables employees to query internal knowledge bases (policies, regulations, SOPs) with accurate, source-cited answers.

**Techniques:**
- Document chunking and embedding (OpenAI, sentence-transformers)
- Vector database storage (Chroma, Pinecone, Weaviate)
- Hybrid retrieval (dense + sparse)
- Re-ranking with cross-encoders
- LLM answer generation with citation grounding
- Evaluation with RAGAS metrics

**Business Impact:**
- 40% reduction in FOIA response timelines
- Self-service policy lookup for employees
- Reduced legal research time
- Improved compliance adherence through accessible SOPs

**Files:**
- `notebooks/01_document_processing.ipynb`
- `notebooks/02_embedding_pipeline.ipynb`
- `notebooks/03_retrieval_evaluation.ipynb`
- `notebooks/04_rag_system.ipynb`
- `src/document_loader.py`
- `src/embedding_engine.py`
- `src/retriever.py`
- `src/rag_pipeline.py`
- `src/evaluate_rag.py`
- `dashboard/knowledge_base_ui.py`

**Status:** 🔧 In development
