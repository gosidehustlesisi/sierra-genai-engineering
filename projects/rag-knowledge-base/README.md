# RAG Knowledge Base for Scientific Literature

**Production-grade Retrieval-Augmented Generation (RAG) system** for searching and synthesizing answers from 2,794 real ArXiv abstracts across machine learning, NLP, computer vision, and data science.

> Architecture mirrors enterprise knowledge base deployments: query transformation → HyDE embedding → FAISS retrieval → cross-encoder re-ranking → LLM answer synthesis with citations.

---

## Architecture

```
User Query
    ↓
Query Transformation (expansion, decomposition)
    ↓
HyDE — Generate hypothetical document
    ↓
FAISS Similarity Search (all-MiniLM-L6-v2 embeddings)
    ↓
Cross-Encoder Re-ranking (ms-marco-MiniLM-L-6-v2)
    ↓
Answer Synthesis with Source Citations
    ↓
Generated Answer + Ranked Sources
```

---

## Features

| Component | Implementation | Purpose |
|------------|----------------|---------|
| **Embeddings** | SentenceTransformers `all-MiniLM-L6-v2` | Dense vector representations |
| **Vector Store** | FAISS (IndexFlatIP) | Cosine similarity search |
| **Query Transform** | LLM-based expansion + decomposition | Vocabulary mismatch fix |
| **HyDE** | Hypothetical document embedding | Answer-as-query retrieval |
| **Re-ranking** | Cross-encoder + LLM-as-judge | Precision optimization |
| **Generator** | Cited answer synthesis | Grounded, attributable output |
| **Dashboard** | Streamlit | Interactive Q&A interface |

---

## Data

- **Source:** `export.arxiv.org/api/query` (live API)
- **Size:** 2,794 unique papers
- **Queries used:** machine learning, NLP, deep learning, computer vision, data science, reinforcement learning, neural networks, transformer, generative AI, large language models
- **Metadata tracked:** title, authors, categories, published date, URL
- **Chunking:** 768-char chunks with 128-char overlap for long abstracts

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Build Vector Store (one-time)

```bash
python src/embeddings.py
```

This encodes all abstracts and builds the FAISS index (~2-3 minutes).

### 3. Run Pipeline Test

```bash
python src/rag_pipeline.py
```

### 4. Launch Dashboard

```bash
streamlit run dashboard.py
```

---

## Project Structure

```
rag-knowledge-base/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/              # ArXiv corpus (gitignored)
│   ├── vector_store/     # FAISS index + metadata (gitignored)
│   └── sample/           # 50-paper demo set
├── src/
│   ├── download_corpus.py    # ArXiv API downloader
│   ├── embeddings.py         # Vector store builder
│   ├── retriever.py          # Query transform + HyDE + retrieval
│   ├── reranker.py           # Cross-encoder re-ranking
│   ├── generator.py          # Answer synthesis
│   └── rag_pipeline.py       # End-to-end orchestration
├── notebooks/
│   ├── 01_corpus_analysis.ipynb
│   └── 02_rag_evaluation.ipynb
├── dashboard.py
└── screenshots/
```

---

## Pipeline API

```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(
    use_reranker=True,       # Cross-encoder precision boost
    use_hyde=True,           # Hypothetical document embedding
    use_llm_transform=True,  # Query expansion/decomposition
    retrieval_k=20,          # First-stage candidates
    final_k=5                # Sources for synthesis
)

result = pipeline.query("What are the latest transformer architectures?")
print(result.answer)
for src in result.sources:
    print(f"[{src['relevance_score']}] {src['title']}")
```

---

## Technical Details

### Query Transformation
- **Expansion:** LLM generates 3 semantically equivalent variants
- **Decomposition:** Complex queries split into 2-3 sub-questions
- **Fallback:** Rule-based synonym expansion when LLM unavailable

### HyDE (Hypothetical Document Embedding)
- Generates a hypothetical answer passage using LLM
- Embeds the hypothetical document instead of the raw query
- Combined with query embeddings via weighted averaging (α=0.6)

### Re-ranking
- **Cross-encoder:** `ms-marco-MiniLM-L-6-v2` scores query-document pairs
- **LLM-as-judge:** Point-wise relevance scoring (0-10 scale)
- Combined score: 70% cross-encoder + 30% original cosine similarity

### Answer Synthesis
- Citations in `[1]`, `[2]` format linked to sources
- Instructed to use ONLY provided context
- Optional iterative refinement (2-pass generation)

---

## Performance

| Metric | Value |
|--------|-------|
| Corpus size | 2,794 papers |
| Chunks | 6,201 (768-char with 128-char overlap) |
| Embedding model | 384-dim all-MiniLM-L6-v2 |
| Retrieval | ~50-150ms |
| Full pipeline | ~500ms-2s (with LLM generation) |

---

## Sources & Citations

- **ArXiv API:** https://export.arxiv.org/api/query
- **Embeddings:** SentenceTransformers (Reimers & Gurevych, 2019)
- **HyDE:** Gao et al. (2022) — "Precise Zero-Shot Dense Retrieval without Relevance Labels"
- **Cross-encoder:** ms-marco-MiniLM-L-6-v2 (Hugging Face)
- **FAISS:** Johnson et al. (2019) — Facebook AI Similarity Search

---

## License

MIT — Built for educational and research purposes.
