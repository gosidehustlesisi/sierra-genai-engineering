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
| 4 | **AI-Ready MLOps** | ML Infrastructure | Model Registry, Drift Detection, A/B Testing, Auto-Retraining | Census ACS + BLS + arXiv APIs | 3 production models | ✅ Complete |
| 5 | **LLM Document Classification** | Enterprise NLP | TF-IDF + Random Forest, Logistic Regression, Confidence Scoring | ArXiv + PubMed + Wikipedia APIs | 991 documents | ✅ Complete |
| 6 | **RAG Knowledge Base** | Retrieval-Augmented Generation | FAISS Indexing, Cross-Encoder Reranking, Semantic Search | arXiv API (8 topic queries) | 2,651 abstracts | ✅ Complete |

**Total**: 6 production projects | 12+ notebooks | 4,154 real documents/records | 0 synthetic data

---

## 📊 Real Data Sources

### arXiv API (Projects 1, 4, 5, 6)
- **Source**: `http://export.arxiv.org/api/query`
- **Categories**: cs.LG, cs.AI, cs.CL, cs.CV, stat.ML
- **License**: arXiv.org perpetual, nonexclusive license
- **What it is**: Live feed of 2,000+ ML/AI/NLP research abstracts with full metadata

### SCOTUS Opinions (Project 2)
- **Source**: Supreme Court of the United States (public domain)
- **License**: U.S. Government Works — Public Domain
- **What it is**: 15 landmark majority opinions — real legal texts, not summaries

### PubMed / Clinical Trials (Projects 3, 5)
- **Source**: PubMed E-utilities (`eutils.ncbi.nlm.nih.gov`)
- **License**: Public domain / open access
- **What it is**: Immunotherapy drug trial outcomes, cancer biomarker data, epidemiology records

### Wikipedia API (Project 5)
- **Source**: MediaWiki REST API (`en.wikipedia.org/api/rest_v1/`)
- **License**: CC BY-SA 3.0
- **What it is**: Legal and financial encyclopedia articles for document classification corpus

### U.S. Census + BLS (Project 4)
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
| **Matplotlib / Seaborn** | Static charts |
| **Plotly** | Interactive visualizations |
| **Streamlit** | Dashboards (MLOps monitor, RAG search, document classifier) |

---

## 🏗️ Architecture

### System Overview
```
Data Ingestion Layer:
  arXiv API → Document Store
  PubMed API → Document Store
  Wikipedia API → Document Store
  Census ACS → Feature Store
  BLS API → Feature Store

NLP Processing Layer:
  Document Store → TF-IDF Vectorization
  Document Store → sentence-transformers Embeddings
  Document Store → spaCy Tokenization

Model Layer:
  TF-IDF → Random Forest Classifier
  TF-IDF → Logistic Regression
  Embeddings → FAISS Index
  Embeddings → Cross-Encoder Reranker

MLOps Layer:
  Classifiers → Model Registry (SQLite)
  Predictions → Drift Detector (KS + PSI)
  Models → A/B Test Router

Application Layer:
  Model Registry → Streamlit Dashboard
  FAISS Index → RAG Search UI
  Classifiers → Document Classifier
```

---

## 📁 Project Structure

```
sierra-genai-engineering/
│
├── 📄 README.md                          ← You are here
├── 📄 LICENSE                            ← MIT License
│
├── 🔧 projects/
│   ├── arxiv-abstracts/                  # Project 1
│   │   ├── notebooks/
│   │   │   └── arxiv_analysis.ipynb
│   │   ├── src/
│   │   ├── data/
│   │   ├── figures/
│   │   └── README.md
│   │
│   ├── scotus-opinions/                  # Project 2
│   │   ├── notebooks/
│   │   │   └── scotus_analysis.ipynb
│   │   ├── data/
│   │   ├── figures/
│   │   └── README.md
│   │
│   ├── pubmed-research/                  # Project 3
│   │   ├── notebooks/
│   │   │   └── pubmed_analysis.ipynb
│   │   ├── data/
│   │   ├── figures/
│   │   └── README.md
│   │
│   ├── ai-ready-mlops/                   # Project 4
│   │   ├── src/
│   │   │   ├── retrain_pipeline.py
│   │   │   ├── drift_detector.py
│   │   │   └── model_registry.py
│   │   ├── notebooks/
│   │   ├── dashboard.py
│   │   └── README.md
│   │
│   ├── llm-document-classification/      # Project 5
│   │   ├── src/
│   │   │   ├── download_documents.py
│   │   │   ├── train_classifier.py
│   │   │   └── evaluate_model.py
│   │   ├── notebooks/
│   │   ├── dashboard.py
│   │   └── README.md
│   │
│   └── rag-knowledge-base/               # Project 6
│       ├── src/
│       │   ├── download_corpus.py
│       │   ├── embeddings.py
│       │   ├── retriever.py
│       │   └── rag_pipeline.py
│       ├── notebooks/
│       ├── dashboard.py
│       └── README.md
│
└── 📊 portfolio-report.pdf               ← Executive summary
```

---

## 🚀 Quick Start

### For Recruiters & Hiring Managers

1. **Executive Summary**: See `portfolio-report.pdf` for business impact metrics
2. **Notebook Demos**: Each project has executable notebooks with real outputs
3. **Live Dashboards**: Streamlit apps for MLOps monitor, RAG search, and document classifier

### For Technical Reviewers

```bash
# Clone the repository
git clone https://github.com/gosidehustlesisi/sierra-genai-engineering.git
cd sierra-genai-engineering

# Install dependencies
pip install -r requirements.txt

# Run any project
jupyter notebook projects/arxiv-abstracts/notebooks/arxiv_analysis.ipynb

# Or launch a dashboard
streamlit run projects/llm-document-classification/dashboard.py
streamlit run projects/rag-knowledge-base/dashboard.py
streamlit run projects/ai-ready-mlops/dashboard.py
```

---

## 📊 Quick Stats

| Metric | Count |
|--------|-------|
| **Projects** | 6 |
| **NLP Pipelines** | 4 |
| **RAG Systems** | 1 |
| **MLOps Frameworks** | 1 |
| **Live Data APIs** | 5 |
| **Documents Analyzed** | 4,154+ |
| **Classification Accuracy** | 89.45% (Logistic Regression) |
| **RAG Retrieval Latency** | 54ms (top-10 on 2,651 docs) |
| **Models Versioned** | 3 (with drift detection + A/B testing) |
| **Dashboards** | 3 Streamlit apps |

---

## 🎯 Brand Positioning

### Why This Portfolio?

Most GenAI portfolios stop at "ChatGPT wrapper" demos. This portfolio demonstrates production NLP infrastructure — from data ingestion through model registry to deployment monitoring.

**The Focus:**
- **"Real Data"** = Every document comes from a live API or public domain source
- **"Production"** = MLOps pipeline with drift detection, A/B testing, auto-retraining
- **"Measurable"** = Classification accuracy, retrieval latency, model versioning metrics

### Target Audience

- **Primary**: Hiring managers evaluating NLP/GenAI engineering candidates
- **Secondary**: Technical leads assessing MLOps and RAG implementation depth
- **Tertiary**: Fellow practitioners seeking reference implementations

---

## 🔗 External Links

| Platform | URL |
|----------|-----|
| 💻 **Portfolio Website** | [e3-ai.com](https://e3-ai.com) |
| 🐙 **GitHub** | `https://github.com/gosidehustlesisi/sierra-genai-engineering` |
| 💼 **LinkedIn** | `https://linkedin.com/in/sierran` |
| 🌐 **Company** | [e3-ai.com](https://e3-ai.com) |

---

## 📝 License

All code, notebooks, and documentation are licensed under the **MIT License** unless otherwise specified.

> *"The best time to build GenAI systems was yesterday. The second best time is now."*
> — **Sierra Napier, The AI Architect**

---

**Last Updated**: May 2026 | **Status**: Production-Ready | **Version**: 2.0
