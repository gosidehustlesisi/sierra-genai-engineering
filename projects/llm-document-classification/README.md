# llm-document-classification

## 🎯 Real-World Context
Document classification pipeline for multi-domain text (legal, scientific, medical, financial). Architecture mirrors production NLP systems deployed for municipal service request triage — specifically the CapSTAT / RatSTAT methodology used by the District of Columbia to automatically classify, route, and prioritize 311 service requests across agencies (DCRA, DOH, DPW, DDOT).

This project demonstrates how the same multi-class text classification + confidence scoring approach scales from civic operations (rodent complaints, illegal dumping, parking violations) to enterprise document intelligence (scientific abstracts, medical literature, legal opinions, financial filings).

## 💼 Business Problem
Organizations process thousands of unstructured documents daily — research papers, medical abstracts, legal opinions, SEC filings — with manual triage that takes 24-72 hours per batch. Delayed classification creates:
- Missed insights in competitive intelligence (financial docs)
- Delayed legal discovery (case law review)
- Slowed research synthesis (scientific literature)
- Inefficient policy triage (public health abstracts)

**Production Parallel:** DC's CapSTAT program reduced rodent complaint response coordination time by 35% through automated classification and geospatial targeting. This pipeline replicates that architecture for general document domains.

## 🔧 Technical Solution
- **Data Sources:** Live public APIs — no synthetic data
  - [ArXiv API](https://arxiv.org/help/api/index) — scientific/technical/financial abstracts
  - [PubMed E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25500/) — medical/health abstracts
  - [Wikipedia REST API](https://www.mediawiki.org/wiki/API:Main_page) — legal/financial encyclopedia articles
- **Stack:** Python 3.10+, Pandas, scikit-learn, spaCy, Streamlit, transformers (optional)
- **Pipeline:** Live API Extract → Clean → TF-IDF Vectorize → Ensemble Classify → Evaluate → Dashboard
- **Key Features:**
  - Multi-class text classification (4 categories: scientific, medical, legal, financial)
  - Confidence scoring per prediction
  - TF-IDF + Random Forest baseline (explainable, fast)
  - Optional DistilBERT fine-tuning (cutting-edge LLM skills)
  - Interactive Streamlit dashboard

## 📊 Results
- **Corpus:** 968 real documents from 3 live APIs
  - ArXiv: 455 scientific/financial abstracts
  - PubMed: 414 medical abstracts
  - Wikipedia: 99 legal/financial articles
- **TF-IDF Features:** 5,000 (unigrams + bigrams)
- **Classification Accuracy:**
  - Logistic Regression: **91.24%** (F1-macro: 0.887)
  - Random Forest: **86.08%** (F1-macro: 0.798)
- **Cross-Validation (5-fold):**
  - Logistic Regression: **91.12%** (+/- 0.031)
  - Random Forest: **86.06%** (+/- 0.025)
- **Per-Class F1 (Logistic Regression):**
  - Medical: **0.976** | Scientific: **0.902** | Legal: **0.941** | Financial: **0.731**
- **Per-Class F1 (Random Forest):**
  - Medical: **0.958** | Scientific: **0.857** | Legal: **0.875** | Financial: **0.500**
- **Pipeline:** End-to-end orchestration with one command
- **Dashboard:** Interactive Streamlit demo with live predictions
- **Notebooks:** Fully executed EDA + modeling notebooks with real output

## 🚀 How to Run

```bash
# Clone and setup
git clone https://github.com/gosidehustlesisi/sierra-genai-engineering.git
cd sierra-genai-engineering/projects/llm-document-classification
pip install -r requirements.txt

# Run full pipeline (downloads live data + trains + evaluates)
python src/run_pipeline.py

# Or run steps individually
python src/download_documents.py   # Fetch live corpus
python src/clean_data.py           # Text cleaning
python src/feature_engineering.py  # TF-IDF + label encoding
python src/train_classifier.py     # Train RF + LR + optional DistilBERT
python src/evaluate_model.py       # Metrics + confusion matrices

# Launch dashboard
streamlit run dashboard.py
```

## 📁 File Structure
```
llm-document-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/              # Real downloaded docs (gitignored)
│   ├── processed/        # Features + labels (gitignored)
│   └── sample/           # 100-row demo CSV for README
├── src/
│   ├── download_documents.py    # Live API client (ArXiv + PubMed + Wikipedia)
│   ├── clean_data.py            # Text cleaning pipeline
│   ├── feature_engineering.py # TF-IDF / embeddings
│   ├── train_classifier.py      # Model training (RF + LR + DistilBERT)
│   ├── evaluate_model.py        # Metrics + confusion matrix plots
│   └── run_pipeline.py          # End-to-end orchestration
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   └── 02_modeling.ipynb        # Model development
├── dashboard.py                 # Streamlit interactive demo
├── models/                      # Saved models (gitignored)
├── reports/                     # Metrics + plots (gitignored)
└── screenshots/                 # PNG images for README
```

## 🏛️ Production Parallels
The architecture mirrors production NLP systems deployed for:
- **Municipal 311 triage** (CapSTAT/RatSTAT) — classify → route → prioritize
- **Legal discovery** — classify case law by domain and relevance
- **Scientific literature monitoring** — auto-categorize preprints for research teams
- **Financial compliance** — triage SEC filings and regulatory documents
- **Healthcare policy** — classify PubMed abstracts for evidence synthesis

## 📄 License
MIT — Open source for portfolio demonstration

---
**About the Author:** Sierra Napier, MPA — Senior Data & AI Leader building enterprise analytics and AI solutions across government, transportation, and financial sectors. [LinkedIn](https://linkedin.com/in/sierran)
