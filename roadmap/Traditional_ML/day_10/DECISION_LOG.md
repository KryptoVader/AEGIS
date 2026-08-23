# Day 10 — Decision Log: Capstone Customer Lifetime Value (CLTV) & Revenue Valuation

## 1. Business & Client Understanding
- **Client**: Executive Marketing & Customer Strategy Division of a Global E-Commerce Retailer.
- **Business Goal**: Predict future Customer Lifetime Value (`cltv_revenue`) across $25,000$ active customers to optimize Customer Acquisition Cost (CAC) bidding, allocate VIP loyalty rewards, and deploy proactive retention campaigns for high-value accounts.
- **Problem Type**: Heavy-Tail Continuous Tweedie/Gamma Regression.
- **Target Variable**: `cltv_revenue` (Continuous lifetime spending value, non-negative, heavily right-skewed, range: $\$10.00\text{--}\$22,756.33$).
- **Dataset Scale**: $25,000$ customer profiles, $12$ features.

---

## 2. Initial Observations & Data Hygiene
- **Data Shape**: $25,000$ rows, $12$ columns.
- **Missing Values**: $0$ missing values.
- **Duplicates**: $0$ duplicate rows.
- **Target Distribution**: Heavy-tail right-skewed revenue curve ($Mean = \$1,175.44, Max = \$22,756.33$).

---

## 3. Exploratory Data Analysis (EDA) Findings
- **`frequency_orders`**: #1 strongest overall positive driver of repeat lifetime customer revenue.
- **`is_vip_member`**: High positive correlation with total spending ($2.2\times$ revenue multiplier).
- **`discount_usage_pct`**: High discount dependence erodes net lifetime customer margins (-40% margin penalty).
- **`customer_service_tickets`**: High complaint rates correlate with customer churn (-12% per ticket penalty).

---

## 4. Feature Engineering & Target Transformation
- **Target Transformation**: `y_log = np.log1p(df['cltv_revenue'])`.
  - Converts right-skewed revenue distribution into a smooth symmetric bell curve.
  - Optimizing MSE on $y_{\text{log}}$ is equivalent to optimizing **RMSLE (Root Mean Squared Logarithmic Error)** on the dollar scale.
- **Categoricals**: `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')` applied to `country`, `device_type`, `acquisition_channel`.
- **Numericals**: `StandardScaler()` applied to `recency_days`, `frequency_orders`, `avg_order_value`, `discount_usage_pct`, `customer_service_tickets`, `days_as_member`.

---

## 5. Model Comparison Leaderboard (One-by-One Tournament)

| Model Architecture | 5-Fold CV / Test $R^2$ Score | Real-World MAE Error ($) | Notes |
| :--- | :---: | :---: | :--- |
| **Linear Regression (Ridge)** | $0.9062$ ($90.62\%$) | $\$245.30$ | Linear lower bound baseline. |
| **SGD Regressor** | $0.9011$ ($90.11\%$) | $\$248.10$ | Stochastic Gradient Descent. |
| **Random Forest Regressor** | $0.9663$ ($96.63\%$) | $\$112.40$ | Bagging ensemble of 200 trees. |
| **GPU XGBoost (Default)** | $0.9702$ ($97.02\%$) | $\$98.60$ | Hardware accelerated boosting. |
| **HistGradientBoosting (Default)** | $0.9734$ ($97.34\%$) | $\$86.20$ | Native binned leaf-wise boosting. |
| **OPTUNA TUNED GPU XGBOOST** | **$\mathbf{0.97337}$** ($97.34\%$) 🏆 | **$\mathbf{\$81.90}$** 🏆 | **GRAND FINALE CAPSTONE CHAMPION!** |

---

## 6. Optuna Bayesian Optimization Results (GPU XGBoost)
- **Objective Function**: Optimized 5-Fold CV $R^2$ score over 10 Bayesian TPE trials on GPU XGBoost.
- **Optuna Best Parameters Discovered**:
  - `n_estimators`: **`568`**
  - `learning_rate`: **`0.0624`** (Optimal step size)
  - `max_depth`: **`5`** (Prevents deep tree overfitting)
  - `subsample`: **`0.8215`**
  - `colsample_bytree`: **`0.7376`**
  - `reg_alpha`: **`3.125e-8`**
  - `reg_lambda`: **`3.5506`** (L2 regularization)
- **Champion $R^2$ Score**: **`0.97337` ($97.34\%$ variance explained!)**

---

## 7. 10-Day AEGIS ML Immersion Retrospective

| Day | Domain & Challenge | Core Technique Mastered | Benchmark Result |
| :---: | :--- | :--- | :---: |
| **Day 1** | Ames Housing Price Valuation | Log Target Transform, Spatial Features | CV Log-RMSE $= 0.1205$ |
| **Day 2** | Telecom Customer Churn | Step Functions, 2D Contour KDEs, GPU XGBoost | $F_1 = 0.9349, \text{ROC-AUC} = 0.9388$ |
| **Day 3** | Bank Marketing Lead Scoring | Target Leakage Removal, Threshold $\tau^*$ | $4\times$ Precision Boost ($11\% \to 46\%$) |
| **Day 4** | Hotel Bookings (119k Scale) | 5-Pillar EDA, Simpson's Paradox ($97.2\%$) | $\text{ROC-AUC} = 0.9489, F_1 = 0.8311$ |
| **Day 5** | Urban Bike Demand Reg. | Cyclic Sine/Cosine Time Encoding | $R^2 = \mathbf{96.07\%}, \text{MAE} = 22.37$ bikes/hr |
| **Day 6** | Wine Quality Multi-Class | Ordinal Reg. + Enology Chemistry Ratios | Accuracy $= 77.94\%$ |
| **Day 7** | Credit Risk Underwriting | Stacking Ensembles, Calibration, SHAP | Precision $= \mathbf{64.57\%}, \text{SOTA ROC-AUC} = 0.7855$ |
| **Day 8** | Store Sales Time-Series | Lags ($t-1, t-7, t-28$), Shift(1) Safeguard | $R^2 = \mathbf{89.02\%}$ (2024 Test Set) |
| **Day 9** | Extreme Fraud Detection | PR-AUC, IsolationForest, GPU XGBoost $\tau^*$ | $\text{PR-AUC} = 0.8749, \text{Precision} = 95.2\%$ |
| **Day 10** | Customer Lifetime Value (CLTV) | Optuna Bayesian Tuning, Heavy-Tail Tweedie | $R^2 = \mathbf{97.31\%}$ 🏆 CAPSTONE |

---

## 8. Final Executive Strategy & CAC Bidding Recommendations
1. **CAC Bidding Rules**: Dynamically adjust Google/Facebook ad acquisition bids based on predicted CLTV segment (bid up to $\$85.00$ for predicted top $10\%$ CLTV customers; cap bids at $\$12.00$ for predicted low-tier accounts).
2. **Proactive VIP Concierge Routing**: Automatically route any customer with predicted $\text{CLTV} > \$2,500.00$ to high-priority VIP customer service queues to eliminate service friction penalties.
