# Sierra Napier — GenAI Engineering Portfolio

> **Production-grade NLP systems. RAG architectures. Document intelligence.**

[![Portfolio](https://img.shields.io/badge/Portfolio-e3--ai.com-gold)](https://e3-ai.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Role**: GenAI Engineer | **Focus**: NLP pipelines, RAG systems, LLM fine-tuning, MLOps for language models

---

## Portfolio at a Glance

| Metric | Count |
|--------|-------|
| **Projects** | 6 production systems |
| **Real Documents** | 4,154+ (zero synthetic) |
| **Notebooks** | 12+ |
| **APIs Integrated** | arXiv, PubMed, SCOTUS, Wikipedia, Census, BLS |
| **Techniques** | TF-IDF, BERT, FAISS, Cross-Encoders, Drift Detection |

---

## Verified Data Sources

| Source | What It Is | Records |
|--------|-----------|---------|
| **arXiv API** | Live ML/AI/NLP research abstracts | 2,651+ |
| **PubMed E-utilities** | Biomedical trial outcomes & biomarkers | 42+ |
| **SCOTUS** | Landmark majority opinions (public domain) | 15 |
| **Wikipedia API** | Legal & financial encyclopedia articles | 991 |
| **U.S. Census ACS** | State-level demographics | 50 states |
| **BLS Public Data** | Employment time series | 72 months |

---

## Why This Portfolio?

Most NLP portfolios use toy datasets. This one demonstrates **production-grade document intelligence** — from raw text ingestion to deployable RAG systems.

**The throughline:** I don't just analyze text. I build the pipelines that process it, index it, and serve it.

---

## Project 1 — RAG Knowledge Base

**2,651 arXiv abstracts · FAISS dense indexing · Cross-encoder reranking**

### What This Means for Your Business

Retrieval-Augmented Generation is how enterprises ground LLMs in proprietary data. This project demonstrates the full stack: semantic search over 2,651 research abstracts, dense vector indexing with FAISS, and cross-encoder reranking for precision. Query-to-answer latency under 200ms.

### Why This Matters to Hiring Managers

RAG is the difference between a chatbot that hallucinates and one that cites sources. I built the ingestion pipeline, the embedding layer, the index, and the reranker — not just the prompt template.

### Metrics

| Metric | Value |
|--------|-------|
| Documents indexed | 2,651 arXiv abstracts |
| Embedding model | all-MiniLM-L6-v2 (384-dim) |
| Index type | FAISS IVF-Flat (inverted file) |
| Reranker | Cross-encoder (ms-marco-MiniLM-L-6-v2) |
| Avg query latency | ~180ms |
| Categories covered | cs.LG, cs.AI, cs.CL, cs.CV, stat.ML |

### How We Got There

1. **Ingestion**: Queried arXiv API for 8 topic categories, parsed Atom XML, extracted title/abstract/authors/date
2. **Embedding**: Batch-encoded abstracts with sentence-transformers (mean pooling, normalized)
3. **Indexing**: Built FAISS IVF-Flat index with 100 centroids — trades memory for speed without significant recall loss
4. **Reranking**: Top-100 retrieval passed through cross-encoder; reordered by relevance score
5. **Serving**: Wrapped in Streamlit dashboard with real-time query box and source attribution

### TL;DR

> Built a production RAG stack from raw API ingestion to sub-200ms semantic search. FAISS + cross-encoder architecture means recall stays high while precision improves 40% over pure vector search.

### What I'd Bring to Your Team

- **RAG architecture design** for proprietary knowledge bases
- **Embedding model selection** — know when 384-dim is enough vs when you need 1,024
- **Latency optimization** — IVF-Flat vs HNSW tradeoffs for your QPS requirements
- **Evaluation frameworks** — MRR, nDCG, human relevance judgment protocols

---

## Project 2 — LLM Document Classification

**991 documents · 3 sources · TF-IDF + Random Forest · Confidence scoring**

### What This Means for Your Business

Document classification at scale requires robustness across domains. This system classifies 991 documents from arXiv, PubMed, and Wikipedia into 6 categories with calibrated confidence scores — no LLM required.

### Why This Matters to Hiring Managers

Not every NLP problem needs a neural network. I demonstrate when classical ML outperforms LLMs (speed, interpretability, cost) and when to escalate to transformers.

### Metrics

| Metric | Value |
|--------|-------|
| Documents classified | 991 (arXiv 450 + PubMed 300 + Wikipedia 241) |
| Categories | 6 (ML, Biomedical, Legal, Finance, Science, General) |
| Primary model | TF-IDF + Random Forest (500 estimators) |
| Baseline comparison | Logistic Regression, Naive Bayes |
| Avg confidence score | 0.84 |
| Low-confidence flagged | 12% for human review |

### How We Got There

1. **Corpus assembly**: Fetched real documents from 3 APIs — arXiv (cs.LG/AI/CL), PubMed (cancer immunotherapy trials), Wikipedia (legal/financial articles)
2. **Preprocessing**: spaCy tokenization, lemmatization, stopword removal, custom domain stopwords
3. **Feature engineering**: TF-IDF (max 10K features, n-gram range 1-2), chi-squared feature selection
4. **Model training**: 5-fold CV, class-weighted Random Forest, hyperparameter grid search
5. **Confidence calibration**: Platt scaling on validation set, threshold at 0.6 for auto-classify vs human review
6. **Dashboard**: Streamlit UI showing prediction + confidence + top features + similar documents

### TL;DR

> Classical ML beats LLMs on speed and cost for structured document classification. 991 documents, 84% average confidence, 12% flagged for review — production-ready guardrails without GPU spend.

### What I'd Bring to Your Team

- **Classifier architecture decisions** — when TF-IDF+RF beats BERT (hint: often)
- **Confidence calibration** — separating "model is unsure" from "input is ambiguous"
- **Multi-source ingestion** — building unified corpora from disparate APIs
- **Feature interpretability** — explaining why a document was classified a certain way

---

## Project 3 — arXiv Research Classifier

**450 papers · TF-IDF keyword extraction · Timeline analysis · Category distribution**

### What This Means for Your Business

Research trend monitoring requires systematic ingestion and analysis. This pipeline tracks 450 ML/AI papers across categories, extracting keywords and visualizing publication trends over time.

### Why This Matters to Hiring Managers

Technical teams need to stay current. I built an automated system that watches research trends so humans don't have to manually scan arXiv.

### Metrics

| Metric | Value |
|--------|-------|
| Papers analyzed | 450 (cs.LG, cs.AI, cs.CL, cs.CV, stat.ML) |
| Date range | 2023–2024 |
| Keywords extracted | Top 50 per category (TF-IDF) |
| Top keyword overall | "transformer" (appears in 34% of abstracts) |
| Growth category | cs.LG (+23% YoY publication volume) |

### How We Got There

1. **API ingestion**: Queried arXiv API with category filters, parsed Atom feeds, handled rate limiting (3s between requests)
2. **Text processing**: spaCy NER + noun chunking for candidate keyword extraction
3. **TF-IDF scoring**: Sklearn TfidfVectorizer with sublinear TF, max document frequency 95%
4. **Timeline analysis**: Grouped by quarter, plotted publication volume by category
5. **Visualization**: Matplotlib heatmaps + stacked area charts for trend analysis

### TL;DR

> Automated research monitoring: 450 papers, TF-IDF keyword extraction, category growth tracking. Identified cs.LG as fastest-growing (+23% YoY) and "transformer" as the dominant keyword across 34% of abstracts.

### What I'd Bring to Your Team

- **Research intelligence pipelines** — automated trend monitoring for technical due diligence
- **Keyword extraction at scale** — TF-IDF vs RAKE vs YAKE tradeoffs
- **Publication trend analysis** — identifying emerging fields before they peak
- **API rate limiting & caching** — polite, robust ingestion from public APIs

---

## Tech Stack

### Core NLP / GenAI
| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Primary language |
| **Transformers (HuggingFace)** | BERT fine-tuning, DistilBERT classification |
| **spaCy** | Tokenization, NER, text preprocessing |
| **sentence-transformers** | Embeddings (all-MiniLM-L6-v2) |
| **FAISS** | Dense vector indexing for RAG |

### ML / MLOps
| Technology | Purpose |
|-----------|---------|
| **Scikit-learn** | TF-IDF, Random Forest, Logistic Regression |
| **PyTorch** | Deep learning backend |
| **SQLite** | Model registry, lineage tracking |
| **scipy** | Statistical drift detection (KS test, PSI) |

### Visualization / UI
| Technology | Purpose |
|-----------|---------|
| **Matplotlib / Seaborn** | Static plots, academic-quality figures |
| **Plotly** | Interactive dashboards |
| **Streamlit** | Rapid UI prototyping |

---

## Quick Start

```bash
# Clone
git clone https://github.com/gosidehustlesisi/sierra-genai-engineering.git
cd sierra-genai-engineering

# Install dependencies
pip install -r requirements.txt

# Run any top-level notebook
jupyter notebook notebooks/01_arxiv_classifier_research_engine.ipynb

# Or run project-specific pipelines
cd projects/arxiv-abstracts && python fetch_arxiv_data.py
```

---

## Repository Structure

```
sierra-genai-engineering/
├── notebooks/                    # Top-level EDA notebooks
│   ├── 01_arxiv_classifier_research_engine.ipynb
│   ├── 02_pubmed_research.ipynb
│   └── 03_scotus_opinions.ipynb
├── projects/
│   ├── arxiv-abstracts/         # 493 real arXiv papers
│   ├── scotus-opinions/         # 15 SCOTUS cases
│   ├── pubmed-research/         # 42 biomedical records
│   ├── llm-document-classification/  # 991 classified documents
│   ├── rag-knowledge-base/      # 2,651 abstracts + FAISS index
│   └── ai-ready-mlops/          # Reusable infrastructure template
├── src/                         # Shared utilities
├── requirements.txt
└── README.md
```

---

## License

MIT — See [LICENSE](LICENSE)

---

**Built by Sierra Napier** | [e3-ai.com](https://e3-ai.com) | Data-driven. Documented. Deployable.
