## Project 1: LLM Document Classification

**Context:** Public-sector document classification inspired by DC Government's FOIA automation and RATS (Rodent Abatement Tracking System) using LLMs.

**Dataset:**
- [Federal Register documents](https://www.federalregister.gov/)
- [DC 311 Service Requests](https://opendata.dc.gov/)
- [MuckRock FOIA requests](https://www.muckrock.com/)
- Simulated internal agency documents

**Objective:** Build production-grade LLM pipelines for document classification, entity extraction, and automated routing with >90% accuracy.

**Techniques:**
- BERT fine-tuning for domain-specific classification
- GPT-4 / Claude zero-shot classification with structured outputs
- Prompt engineering with chain-of-thought reasoning
- FastAPI serving with request batching and caching

**Business Impact:**
- 30% reduction in rodent complaint resolution time (RATS)
- 40% reduction in FOIA response timelines
- Automated metadata classification across 7 departments
- 35% faster model deployment via AI-ready environments

**Files:**
- `notebooks/01_document_exploration.ipynb`
- `notebooks/02_bert_finetuning.ipynb`
- `notebooks/03_llm_zero_shot.ipynb`
- `notebooks/04_prompt_engineering.ipynb`
- `src/bert_classifier.py`
- `src/llm_pipeline.py`
- `src/prompt_templates.py`
- `src/serve.py`
- `docker/Dockerfile`

**Status:** 🔧 In development
