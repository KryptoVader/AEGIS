# AEGIS

**AEGIS** is a repository built to bridge the gap between academic tutorials and real-world deployment across **Traditional Machine Learning** and **Deep Learning**.

The repository serves as an applied laboratory log and decision framework for building, validating, and optimizing end-to-end AI systems across complex operational domains.

---

## Motive & Core Philosophy

Standard machine learning and deep learning tutorials rely on clean, toy datasets evaluated on simple accuracy metrics with random train/test shuffles. In real-world enterprise environments, these naive approaches fail due to asymmetric business costs, severe class imbalance, data leakage, complex temporal dynamics, and uncalibrated neural outputs.

AEGIS was created to establish a rigorous, production-first engineering standard for applied AI, grounded in four core principles:

1. **Business-Aligned Evaluation Metrics**: Rejecting naive accuracy in favor of domain-appropriate metrics—such as Precision-Recall AUC (PR-AUC) for extreme fraud ($0.1\%$ prevalence), Root Mean Squared Logarithmic Error (RMSLE) for heavy-tailed spend, and Cost-Weighted Loss Functions.
2. **Strict Data Hygiene & Leakage Audit**: Enforcing zero-leakage protocols by auditing features for post-event data contamination, isolating target variables prior to preprocessing pipelines, and applying `.shift(1)` safeguards on historical rolling window calculations.
3. **Validation Realism**: Reallocating focus from standard random k-fold splits to Out-of-Time (OOT) temporal validation for time-series forecasting, and Stratified K-Fold validation for imbalanced classification.
4. **Interpretability & Neural Calibration**: Guaranteeing that traditional ensembles and deep neural networks produce calibrated, real-world probability estimates (`CalibratedClassifierCV`) accompanied by regulatory compliance explanations (`SHAP`).

---

## Purpose & How This Repository Is Used

This repository is designed as an operational reference library and decision log for machine learning and deep learning engineers:

- **Reference Implementations**: Production-ready Python and C pipelines covering tabular preprocessing (`ColumnTransformer`, `RobustScaler`), model training (GPU-accelerated XGBoost, LightGBM, CatBoost), neural architectures (PyTorch / C-based Deep Learning from scratch), Bayesian hyperparameter optimization (`Optuna`), and multi-model stacking ensembles (`StackingClassifier`).
- **Architectural Decision Logs**: Detailed documentation (`DECISION_LOG.md`) accompanying every domain experiment, capturing problem formulations, exploratory findings, feature engineering rationale, baseline comparison leaderboards, and business deployment rules.
- **Exploratory Data Analysis Framework**: Standardized visual guidelines (`EDA_VISUALIZATION_FRAMEWORK.md`) for diagnosing feature distributions, multi-collinearity, interaction effects, and data quality anomalies.

---
