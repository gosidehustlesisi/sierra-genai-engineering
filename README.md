# Sierra Napier — GenAI Engineering Portfolio

> *"The best time to build AI agents was yesterday. The second best time is now."*
> — **The AI Architect**

**Build Production-Grade GenAI Systems. Deploy Real-World NLP Solutions. Drive Measurable Impact.**

[![Portfolio](https://img.shields.io/badge/Portfolio-e3--ai.com-blue)](https://e3-ai.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Role**: GenAI Engineer | **Focus**: NLP pipelines, RAG systems, LLM fine-tuning, MLOps for language models

---

## 📦 Deliverable Inventory

| # | Project | Domain | Techniques | Real Data Source | Records | Status |
|---|---------|--------|------------|-----------------|---------|--------|
| 1 | **arXiv Abstracts Analysis** | Academic NLP | TF-IDF, Keyword Frequency, Timeline Analysis | arXiv API (cs.LG, cs.AI, cs.CL) | 450 papers | ✅ Complete |
| 2 | **SCOTUS Opinions Analysis** | Legal NLP | Text Length Trends, Topic Distribution, Term Frequency | Supreme Court majority opinions (public domain) | 15 landmark cases | ✅ Complete |
| 3 | **PubMed Research Analysis** | Medical NLP | Response Rate Analysis, Biomarker Volcano Plots, Epidemiology Scatter | PubMed / clinical trial registries | 42 records (20 trials + 12 biomarkers + 10 epidemiology) | ✅ Complete |
| 4 | **LLM Document Classification** | Enterprise NLP | TF-IDF + Random Forest, Logistic Regression, Confidence Scoring | ArXiv + PubMed + Wikipedia APIs | 991 documents | ✅ Complete |
| 5 | **RAG Knowledge Base** | Retrieval-Augmented Generation | FAISS Indexing, Cross-Encoder Reranking, Semantic Search | arXiv API (8 topic queries) | 2,651 abstracts | ✅ Complete |
| — | **AI-Ready MLOps** *(Infrastructure Template)* | ML Infrastructure | Model Registry, Drift Detection, A/B Testing, Auto-Retraining | Census ACS + BLS + arXiv APIs | Plug your own data | 🏗️ Reusable Template |

**Total**: 5 production projects | 12+ notebooks | 4,154 real documents/records | 0 synthetic data

---

## 📊 Real Data Sources

### arXiv API (Projects 1, 4, 5)
- **Source**: `http://export.arxiv.org/api/query`
- **Categories**: cs.LG, cs.AI, cs.CL, cs.CV, stat.ML
- **License**: arXiv.org perpetual, nonexclusive license
- **What it is**: Live feed of 2,000+ ML/AI/NLP research abstracts with full metadata

### SCOTUS Opinions (Project 2)
- **Source**: Supreme Court of the United States (public domain)
- **License**: U.S. Government Works — Public Domain
- **What it is**: 15 landmark majority opinions — real legal texts, not summaries

### PubMed / Clinical Trials (Projects 3, 4)
- **Source**: PubMed E-utilities (`eutils.ncbi.nlm.nih.gov`)
- **License**: Public domain / open access
- **What it is**: Immunotherapy drug trial outcomes, cancer biomarker data, epidemiology records

### Wikipedia API (Project 4)
- **Source**: MediaWiki REST API (`en.wikipedia.org/api/rest_v1/`)
- **License**: CC BY-SA 3.0
- **What it is**: Legal and financial encyclopedia articles for document classification corpus

### U.S. Census + BLS (Infrastructure Template)
- **Census ACS**: `https://api.census.gov/data/2022/acs/acs5` — state-level demographics
- **BLS**: `https://api.bls.gov/publicAPI/v2/timeseries/data/` — employment time series
- **License**: U.S. Government Works — Public Domain

---

## 🛠️ Tech Stack

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

## 📁 Repository Structure

```
sierra-genai-engineering/
├── notebooks/                          # ← Top-level EDA notebooks (NEW)
│   ├── 01_arxiv_classifier_research_engine.ipynb
│   ├── 02_pubmed_research.ipynb
│   └── 03_scotus_opinions.ipynb
├── projects/
│   ├── arxiv-abstracts/
│   │   ├── data/                       # 493 real arXiv papers (CSV + JSON)
│   │   ├── notebooks/                  # Project-specific analysis
│   │   ├── figures/
│   │   └── fetch_arxiv_data.py
│   ├── scotus-opinions/
│   │   ├── data/                       # 15 SCOTUS cases (JSON)
│   │   ├── notebooks/
│   │   └── figures/
│   ├── pubmed-research/
│   │   ├── data/                       # 42 biomedical records (JSON)
│   │   ├── notebooks/
│   │   └── figures/
│   ├── llm-document-classification/
│   │   ├── data/                       # 991 classified documents
│   │   ├── notebooks/
│   │   ├── src/
│   │   └── dashboard.py
│   ├── rag-knowledge-base/
│   │   ├── data/                       # 2,651 abstracts + FAISS index
│   │   ├── notebooks/
│   │   ├── src/
│   │   └── dashboard.py
│   └── ai-ready-mlops/
│       ├── notebooks/
│       ├── src/
│       └── dashboard.py
├── src/                                # Shared utilities
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

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

## 📜 License

MIT — See [LICENSE](LICENSE)

---

**Built by Sierra Napier** | [e3-ai.com](https://e3-ai.com) | Data-driven. Documented. Deployable.
