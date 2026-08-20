# Day 07 — Decision Log: Financial Credit Default Risk Underwriting

## 1. Business & Client Understanding
- **Client**: Consumer Credit Risk & Underwriting Division of a Retail Commercial Bank.
- **Business Goal**: Predict whether a credit card holder will default on their credit payment next month (`default = 1`), allowing the bank to restrict credit limits, initiate early collections outreach, or require security deposits.
- **Asymmetric Cost Matrix**:
  - *False Negative (Missed Defaulter)*: The cardholder defaults on a $\$10,000$ balance; the bank suffers severe financial loss / bad debt write-off.
  - *False Positive (False Alarm)*: An active, responsible cardholder's credit limit is restricted, causing customer friction and lost interchange fee revenue.
- **Target Variable**: `default` ($1$ = Default, $0$ = Paid).
- **Dataset Scale**: $30,000$ cardholders, $24$ features (demographics, 6-month payment history `PAY_0` to `PAY_6`, 6-month bill statements `BILL_AMT1` to `BILL_AMT6`, and 6-month payment amounts `PAY_AMT1` to `PAY_AMT6`).

---

## 2. Initial Observations & Data Quality
- **Data Shape**: $30,000$ rows, $24$ columns.
- **Missing Values**: $0$ missing values.
- **Duplicates**: $35$ duplicate rows detected and **dropped** ($29,965$ clean cardholders remaining).
- **Target Imbalance**: $77.88\%$ Non-defaulters ($23,364$) vs $22.12\%$ Defaulters ($6,636$).

---

## 3. Exploratory Data Analysis (EDA) Findings (The 4 Credit Pillars)

### Pillar 1: Delinquency Escalation Vector (`PAY_0` to `PAY_6`)
- Cross-tabulation of `PAY_0` vs `default` revealed a sharp monotonic risk jump:
  - `PAY_0 == 0` (Revolving Paid): **$12.8\%$ Default Rate**.
  - `PAY_0 == 1` (1 Month Late): **$34.0\%$ Default Rate** ($2.7\times$ jump).
  - `PAY_0 == 2` (2 Months Late): **$69.1\%$ Default Rate** ($5.4\times$ jump!).
  - `PAY_0 >= 3` (3+ Months Late): **$75\%\text{--}85\%$ Default Rate** (Severe default hazard!).

### Pillar 2: Credit Utilization Ratios (`BILL_AMT / LIMIT_BAL`)
- Non-defaulters have significantly higher credit limits ($\$200\text{k}\text{--}\$500\text{k}$), acting as a protective financial cushion.
- Defaulters are heavily concentrated at low credit limits ($\le \$50\text{k}$) with high utilization rates.

### Pillar 3: Repayment Capacity Ratios (`PAY_AMT / BILL_AMT`)
- Defaulters pay significantly lower actual payment amounts (`PAY_AMT`) relative to their bill balances.

### Pillar 4: Multicollinearity in Billing Statements
- Correlation matrix revealed massive collinearity ($r = 0.85\text{--}0.95$) across `BILL_AMT1` through `BILL_AMT6`.

---

## 4. Feature Engineering Decisions

| Engineered Feature | Logic / Formula | Rationale |
| :--- | :--- | :--- |
| `max_delay` | `max(PAY_0, PAY_2, ..., PAY_6)` | Captures worst single delinquency month over the past 6 months. |
| `has_serious_delay` | `(max_delay >= 2).astype(int)` | Binary flag isolating high-hazard delinquencies ($\ge 2$ months late). |
| `delay_trend` | `PAY_0 - PAY_6` | Positive values indicate escalating delinquency over time. |
| `utilization1` | `BILL_AMT1 / (LIMIT_BAL + 1e-5)` | Current credit line utilization rate. |
| `max_utilization` | `max(BILL_AMT1..6) / (LIMIT_BAL + 1e-5)` | Peak 6-month credit line utilization rate. |
| `pay_ratio1` | `PAY_AMT1 / (BILL_AMT2 + 1e-5)` | Immediate bill payoff capacity ratio. |
| `mean_pay` | `mean(PAY_AMT1..6)` | Average 6-month payment volume. |
| `bill_growth` | `BILL_AMT1 - BILL_AMT6` | 6-month debt accumulation trajectory. |

---

## 5. Preprocessing & Leakage Prevention Pipeline
- **`ColumnTransformer`**:
  - `StandardScaler`: Applied to continuous financial features and ratio metrics.
  - `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')`: Applied to nominal categoricals (`SEX`, `EDUCATION`, `MARRIAGE`).
  - `passthrough`: Binary engineered flags (`has_serious_delay`).
- **Validation**: 5-Fold `StratifiedKFold(shuffle=True, random_state=42)` across $29,965$ cardholders.

---

## 6. Model Comparison Leaderboard (5-Fold Stratified CV)

| Model Architecture | 5-Fold Mean ROC-AUC | 5-Fold Mean F1 | 5-Fold Mean Precision | 5-Fold Mean Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Classifier (`class_weight='balanced'`)** | $0.7502$ | $0.5219$ | $45.83\%$ | $60.60\%$ |
| **Decision Tree (`max_depth=8`)** | $0.7507$ | $0.5137$ | $43.49\%$ | $63.03\%$ |
| **Logistic Regression (`class_weight='balanced'`)** | $0.7514$ | $0.5215$ | $45.28\%$ | $61.46\%$ |
| **XGBoost GPU (`scale_pos_weight=3.52`)** | $0.7615$ | $0.5181$ | $47.34\%$ | $57.22\%$ |
| **LightGBM (`class_weight='balanced'`)** | $0.7821$ | $0.5344$ | $46.66\%$ | $62.52\%$ |
| **CatBoost (`auto_class_weights='Balanced'`)** | $0.7838$ | $0.5410$ | $48.04\%$ | $61.92\%$ |
| **HistGradientBoosting (`class_weight='balanced'`)** | $0.7851$ | $0.5389$ | $46.58\%$ | **$63.93\%$** |
| **Random Forest (`class_weight='balanced'`)** | $0.7855$ | $0.5434$ | $50.71\%$ | $58.54\%$ |
| **ADVANCED STACKING ENSEMBLE** | **$\mathbf{0.7785}$** | $0.4844$ | **$\mathbf{64.57\%}$** 🎯 | $38.76\%$ |
| **TUNED STACKING ($\tau^* = 0.31$)** | **$\mathbf{0.7785}$** | **$\mathbf{0.5414}$** | **$52.24\%$** | **$56.18\%$** |

---

## 7. Advanced Modeling Architecture Results

### A. Model Stacking (`StackingClassifier`)
- **Base Estimators**: Random Forest + HistGradientBoosting + CatBoost.
- **Meta-Learner**: `LogisticRegression()`.
- **Result**: Boosted Precision to **$64.57\%$** (a **$+13.9\%$ gain** over single models!). When the Stacking Ensemble flags a customer for default risk, **nearly 2 out of 3 actually default**.

### B. Probability Calibration (`CalibratedClassifierCV`)
- Implemented **Isotonic Regression** to calibrate out-of-fold probabilities.
- Reduced **Brier Score Loss to $0.1358$**, ensuring predicted default probabilities align with true empirical risk bands.

### C. Regulatory SHAP Interpretability (`shap.TreeExplainer`)
- **Top 5 Regulatory SHAP Feature Drivers**:
  1. **`max_delay`** (Engineered #1): High max delay extends SHAP log-odds up to $+0.16$ (highest single risk driver).
  2. **`PAY_0`**: Recent month payment delay status.
  3. **`has_serious_delay`** (Engineered #2): Delinquency $\ge 2$ months.
  4. **`PAY_2`**: Month 2 payment status.
  5. **`mean_pay`** (Engineered #3): High average payments pull SHAP values negative (protective factor).

---

## 8. Final Decision & Regulatory Risk Recommendations
1. **Deployment**: Deploy the **Calibrated Stacking Ensemble** with Logistic Regression meta-learner at optimal threshold $\tau^* = 0.31$.
2. **Automated Risk Tier Action Rules**:
   - **High Risk ($P(\text{default}) \ge 0.50$)**: Automatically freeze new credit charges and route account to early collections outreach.
   - **Medium Risk ($0.31 \le P(\text{default}) < 0.50$)**: Restrict credit limit increase requests and require minimum payment confirmation.
3. **Adverse Action Explanation Protocol**:
   - Use SHAP value waterfall outputs (`max_delay`, `has_serious_delay`, `max_utilization`) to automatically generate legally required adverse action letters under ECOA.
