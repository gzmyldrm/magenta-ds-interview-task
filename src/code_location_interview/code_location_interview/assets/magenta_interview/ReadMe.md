## What’s here (MVP)

- **Data gen / load:** `get_data.py` (given synthetic data)
- **Processing & FE:** `process_raw_data.py` (type fixes, buckets, missing flags, merging, etc).  
- **Split:** `split_train_test.py` (group-aware by customer).  
- **Train:** `train.py` (simple one model setting).  
- **Predict:** `predict.py` (batch predictions on test).  
- **Evaluate:** `eval.py` (baseline upsell rate, Precision@k at 5/10/20%, Recall@k, Lift@k etc.).  
- **Notebook (optional):** `explore.ipynb` for quick visuals that fed the slides.  

---

## What’s stubbed (roadmap)

- **`promoter/`** (placeholder): criteria + hooks to promote a trained model to “current champion.”  
- **`tuning/`** (placeholder): small search utilities for hyper-parameter tuning (e.g., randomized search).  
- **`feature_selection/`** (placeholder): wrappers for filter/embedded selectors + permutation importance.  
- **More training:** extend `train.py` to support additional models, e.g. **XGBoost** and **Logistic Regression**, for baseline comparison.  
- **Tests:** add pytests
- **More Dagster familiarity & integrations (planned):** learn more and explore Dagster’s features and possible integrations such as **MLflow**, **Evidently**, or similar tools for experiment tracking, monitoring, and drift detection.  