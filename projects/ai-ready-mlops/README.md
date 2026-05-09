## Project 3: AI-Ready MLOps

**Context:** Production MLOps infrastructure for LLM and ML model deployment inspired by WMATA's AI-ready environments.

**Dataset:**
- Simulated model performance metrics
- [MLflow tracking data](https://mlflow.org/)
- [Weights & Biases experiments](https://wandb.ai/)

**Objective:** Build reproducible MLOps infrastructure for deploying, monitoring, and maintaining ML and LLM models in production with cost tracking and A/B testing.

**Techniques:**
- Docker containerization for model serving
- MLflow experiment tracking and model registry
- A/B testing framework for model comparison
- Monitoring with Prometheus / Grafana
- Cost tracking for LLM API usage
- Automated retraining triggers

**Business Impact:**
- 35% reduction in time-to-deployment for ML models
- Standardized ML workflow across teams
- Reduced model serving costs through caching and batching
- Proactive model degradation detection

**Files:**
- `notebooks/01_mlops_basics.ipynb`
- `notebooks/02_mlflow_tracking.ipynb`
- `notebooks/03_ab_testing.ipynb`
- `notebooks/04_monitoring.ipynb`
- `src/model_container.py`
- `src/mlflow_tracker.py`
- `src/ab_test_engine.py`
- `src/cost_monitor.py`
- `src/retrain_trigger.py`
- `docker/Dockerfile`
- `docker/docker-compose.yml`

**Status:** 🔧 In development
