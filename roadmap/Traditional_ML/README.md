# AEGIS — Traditional ML Engineering Roadmap

This directory contains the decision logs, exploratory workflows, and pipeline implementations developed across the AEGIS Traditional Machine Learning Roadmap.

---

## Motive & Engineering Principles

The Traditional ML roadmap focuses on solving tabular data challenges under real-world constraints:

1. **Domain-Specific Metric Alignment**: Matching metrics to real business costs (PR-AUC for extreme $0.1\%$ fraud, RMSLE for heavy-tailed spend, Macro F1 for rare-class multi-class distributions).
2. **Leakage Prevention & Validation Realism**: Enforcing Out-of-Time splits for time-series data, stripping post-event features (e.g., call duration, reservation status), and applying `.shift(1)` safeguards to rolling statistics.
3. **Advanced Architecture & Explainability**: Combining GPU-accelerated gradient boosting (XGBoost, LightGBM, CatBoost), Optuna Bayesian hyperparameter optimization, probability calibration (`CalibratedClassifierCV`), and SHAP feature explainability.

---

## Directory Overview

Each module directory contains:
- `data/`: Local dataset storage.
- `day_XX.ipynb`: End-to-end Python notebook implementation.
- `DECISION_LOG.md`: Formal engineering document capturing business context, exploratory findings, baseline leaderboards, feature importances, and operational deployment rules.

Refer to [EDA_VISUALIZATION_FRAMEWORK.md](file:///c:/AEGIS/roadmap/Traditional_ML/EDA_VISUALIZATION_FRAMEWORK.md) for standard visual exploration guidelines.
