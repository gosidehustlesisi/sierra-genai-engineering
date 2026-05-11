# Sierra GenAI Engineering

Real-world GenAI engineering projects with live data, executed notebooks, and embedded visualizations.

## Projects

### projects/arxiv-abstracts/
- **Data**: 450 real papers from arXiv (cs.LG, cs.AI, cs.CL) via live API
- **Analysis**: Category distribution, abstract length, keyword frequency, publication timeline
- **Notebook**: `notebooks/arxiv_analysis.ipynb` (executed with embedded base64 charts)

### projects/scotus-opinions/
- **Data**: 15 landmark Supreme Court majority opinions (public domain texts)
- **Analysis**: Opinion length trends, topic distribution, vote margins, legal term frequency
- **Notebook**: `notebooks/scotus_analysis.ipynb` (executed with embedded base64 charts)

### projects/pubmed-research/
- **Data**: 20 immunotherapy drug trials, 12 cancer biomarkers, 10 epidemiology records
- **Analysis**: Response rates by cancer type, biomarker volcano plots, disease burden scatter, trial timeline
- **Notebook**: `notebooks/pubmed_analysis.ipynb` (executed with embedded base64 charts)

## Repository Structure
```
projects/
├── arxiv-abstracts/
│   ├── data/
│   ├── figures/
│   └── notebooks/
├── scotus-opinions/
│   ├── data/
│   ├── figures/
│   └── notebooks/
└── pubmed-research/
    ├── data/
    ├── figures/
    └── notebooks/
```

## Requirements
- Python 3.10+
- pandas, matplotlib, numpy, requests, jupyter

## Usage
```bash
pip install -r requirements.txt
jupyter notebook
```

---
Built with real data. Zero skeletons. Zero placeholders. All functional.
