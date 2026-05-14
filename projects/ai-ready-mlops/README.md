# AI-Ready MLOps Framework
## Production ML Model Lifecycle Management

**🎯 Real-World Context:** Production ML systems fail silently. This framework demonstrates the operational infrastructure needed to keep ML models reliable — automated retraining, drift detection, A/B testing, and one-click rollback.

**💼 Business Problem:** ML models in production suffer from model decay, undetected drift, rollback nightmares, and no performance visibility. This framework addresses all four.

**🔧 Technical Solution:**
- **Data Sources:**
  - [U.S. Census ACS API](https://api.census.gov/data/2022/acs/acs5) — state-level demographics
  - [BLS Public Data API](https://api.bls.gov/publicAPI/v2/timeseries/data/) — employment time series
  - [arXiv API](http://export.arxiv.org/api/query) — CS paper abstracts
- **Stack:** Python, scikit-learn, SQLite, scipy, Streamlit
- **Pipeline:** Train → Register → Monitor → Detect Drift → Retrain → A/B Test → Promote

**📊 Results:**
- 3 production models versioned with full lineage in SQLite registry
- Automated weekly retraining with 2% regression auto-rejection gate
- Statistical drift detection (KS test + PSI) with alert thresholds
- A/B testing with traffic routing and significance validation
- Inference monitoring: p50/p95/p99 latency, cost per 1K inferences

**🏛️ Production Parallels:** Stripe (fraud model versioning), Netflix (A/B testing), Uber (drift detection)

**🚀 How to Run:**
```bash
pip install -r requirements.txt
python -c "from src.retrain_pipeline import main; main()"
python src/drift_detector.py
streamlit run dashboard.py
```

**📄 License:** MIT

**About the Author:** Sierra Napier, MPA/MPH — AI Architect & Data Science Leader. [LinkedIn](https://linkedin.com/in/sierran)
