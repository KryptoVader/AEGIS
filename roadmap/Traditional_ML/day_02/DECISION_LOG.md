# Day 02 — Decision Log: Telecom Customer Churn Prediction

## 1. Business & Client Understanding
- **Client**: Telecom Service Provider (Customer Retention & Operations Team).
- **Business Goal**: Predict which subscribers are at high risk of canceling their subscriptions (churn) so the retention team can intervene with proactive customer support outreach and targeted offers.
- **Cost/Impact Trade-off**:
  - *False Negative (Missed Churner)*: The customer leaves permanently; customer acquisition cost ($CAC$) to replace them is $5\times\text{–}10\times$ higher than retention cost.
  - *False Positive (False Alarm)*: An unnecessary discount/incentive is offered to a loyal customer (wasted budget/margin loss).
- **Target Variable**: `class` ($1$ = Churned, $0$ = Retained).
- **Imbalance Ratio**: $85.86\%$ Retained ($4,293$), $14.14\%$ Churned ($707$).

---

## 2. Initial Observations & Data Quality
- **Data Shape**: $5,000$ rows, $21$ columns.
- **Missing Values**: $0$ missing values across all columns.
- **Duplicates**: $0$ duplicate rows.
- **Identifiers**: `phone_number` is a unique sequential key (uniform distribution, straight-line ECDF) $\to$ **Dropped** to prevent memorization.

---

## 3. Hypotheses & Domain Inquiries
1. *Frustration Hypothesis*: Customers with high customer service calls are having billing/service issues and are substantially more likely to leave.
2. *Bill Shock Hypothesis*: High daytime usage (which costs $\$0.17/\text{min}$) drives large monthly bills that trigger churn.
3. *International Plan Hazard*: International callers churn at elevated rates due to high international rates or competitor roaming deals.
4. *Voicemail Stickiness*: Customers actively using voicemail have higher switching friction and are less likely to leave.

---

## 4. Exploratory Data Analysis (EDA) Findings (The 4 Pillars)

### Pillar 1: Customer Tenure & Geography
- `account_length`: Completely uniform across churners and non-churners (both center at $\sim 100$ months). Customer tenure alone does not drive churn.
- `state` / `area_code`: Churn rates are broadly uniform across area codes ($408, 415, 510$ all $\sim 14\%$).

### Pillar 2: Value-Add Service Plans (Interaction Effect)
- `international_plan == 1` $\longrightarrow$ **$42.1\%$ Churn Rate** ($4\times$ base rate!).
- `voice_mail_plan == 1` $\longrightarrow$ **$7.7\%$ Churn Rate** (cuts churn risk in half!).
- **4-Way Interaction Breakdown**:
  - `No Plans`: $13.6\%$ Churn.
  - `Intl Only, No Voicemail`: **$44.7\%$ Churn** (Highest hazard cohort!).
  - `Voicemail Only, No Intl`: **$4.7\%$ Churn** (Most loyal cohort!).
  - `Bundled (Both Plans)`: $35.1\%$ Churn (Voicemail provides a $\sim 10\%$ protective cushion).

### Pillar 3: Multi-Daypart Usage & Billing
- **Call Volume vs Call Duration**: `total_day_calls` is centered at $100$ calls for both churners and non-churners (zero predictive signal). But `total_day_minutes` is **bimodal** for churners with a heavy tail at $270+$ minutes.
- **Collinearity**: `total_*_charge` is an exact mathematical multiple of `total_*_minutes` ($r = 1.0$). 
- **Total Spend**: Churners show a distinct high-spend mode at **$\$75\text{–}\$80+$** total monthly bill.

### Pillar 4: Customer Friction (The Cliff Edge)
- Calls $0\text{–}3$: Churn rate is flat at $\sim 10\text{–}12\%$.
- Call $4$: Churn rate jumps sharply to **$45\%$**.
- Calls $5\text{–}6$: Churn rate climbs to **$60\%\text{–}65\%$**.
- Call $9$: $100\%$ churn rate.

---

## 5. Feature Engineering Decisions

| Engineered Feature | Formula / Logic | Rationale |
| :--- | :--- | :--- |
| `total_bill` | Sum of Day + Eve + Night + Intl charges | Captures total monthly financial spend in one consolidated metric. |
| `total_minutes` | Sum of Day + Eve + Night + Intl minutes | Captures total overall usage volume. |
| `day_charge_ratio` | `total_day_charge / total_bill` | Measures fraction of spend from expensive daytime calling ($0.60$ for churners vs $0.50$ for retained). |
| `service_calls_high` | `(number_customer_service_calls >= 4).astype(int)` | Explicitly gives linear models the non-linear step-function cliff. |
| `intl_no_vmail` | `((international_plan == 1) & (voice_mail_plan == 0)).astype(int)` | Flags the highest-risk subscriber segment ($44.7\%$ churn). |

---

## 6. Preprocessing & Leakage Prevention Decisions
- **`ColumnTransformer`**:
  - `StandardScaler`: Applied to all continuous usage features and discrete call counts.
  - `OneHotEncoder`: Applied to `area_code` (and `state`).
  - `passthrough`: Binary flags (`international_plan`, `voice_mail_plan`, `service_calls_high`, `intl_no_vmail`).
- **Zero Leakage**: All transformations fit inside the CV pipeline exclusively on training folds.

---

## 7. Validation Strategy & Metric Selection
- **Validation**: 5-Fold `StratifiedKFold(shuffle=True, random_state=42)` to strictly preserve the $85.9\% / 14.1\%$ class ratio across all folds.
- **Evaluation Metrics**:
  - **F1-Score**: Primary classification metric (harmonic mean of Precision & Recall).
  - **ROC-AUC**: Primary probability ranking metric.

---

## 8. Model Comparison Leaderboard (5-Fold Stratified CV)

| Model Family | Model Architecture | 5-Fold Mean F1 | 5-Fold Mean ROC-AUC | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Linear Baseline** | Logistic Regression (`class_weight='balanced'`) | $0.6203$ | $0.8719$ | Linear log-odds boundary ceiling. |
| **Linear (L2)** | Ridge Classifier (`class_weight='balanced'`) | $0.6285$ | $0.8725$ | L2 regularization adds slight stability. |
| **Linear (L1)** | Lasso Logistic Regression (`class_weight='balanced'`) | $0.6207$ | $0.8719$ | Feature zeroing doesn't resolve non-linear splits. |
| **Single Tree** | Decision Tree (`class_weight='balanced'`) | $0.8451$ | $0.9110$ | **$+0.22$ F1 Jump!** Orthogonal splits capture thresholds. |
| **Bagging** | Random Forest | $0.9023$ | $0.9209$ | Bootstrap aggregation reduces variance past $0.90$ F1. |
| **Bagging** | Bagging Classifier | $0.9060$ | $0.9214$ | Robust ensemble bagging. |
| **Boosting** | CatBoost | $0.9202$ | $0.9229$ | Superb handling of plan interactions. |
| **Boosting** | LightGBM | $0.9160$ | **$0.9272$** | Highest untuned ROC-AUC ranking score. |
| **Boosting** | XGBoost (Untuned) | $0.9145$ | $0.9254$ | Fast GPU histogram boosting. |
| **TUNED CHAMPION** | **HistGradientBoosting (Tuned)** | **$\mathbf{0.9349}$** | **$\mathbf{0.9388}$** | **Top Performer ($+31.4\%$ F1 gain over baseline!)** |
| **TUNED CHAMPION** | **XGBoost GPU (Tuned)** | **$\mathbf{0.9349}$** | **$\mathbf{0.9388}$** | **Tied top performer (GTX 1650 accelerated)** |

---

## 9. Hyperparameter Tuning & Model Explanations
- **Best Parameters for HistGradientBoosting**:
  - `learning_rate`: $0.08$ / $0.1$
  - `max_leaf_nodes`: $31$
  - `min_samples_leaf`: $20$ (prevents leaf over-fitting on small pockets)
  - `l2_regularization`: $1.0$ (stabilizes leaf weights)
  - `class_weight`: None (trees optimize leaf splits natively)

---

## 10. Feature Importance (Permutation Importance Verification)

| Rank | Feature | Permutation Importance Mean | Business Driver |
| :---: | :--- | :---: | :--- |
| **1** | `total_bill` | **$0.1349$** | Financial bill shock / total spend is the #1 churn driver. |
| **2** | `number_customer_service_calls` | **$0.0607$** | Customer friction & dissatisfaction at $\ge 4$ calls. |
| **3** | `international_plan` | **$0.0524$** | International calling risk factor ($42\%$ churn rate). |
| **4** | `number_vmail_messages` | **$0.0293$** | Product stickiness / retention anchor. |
| **5** | `total_intl_calls` | **$0.0233$** | International usage volume. |

---

## 11. Final Decision & Business Recommendations
1. **Model Deployment**: Deploy the tuned **HistGradientBoostingClassifier** ($F1 = 0.935, \text{ROC-AUC} = 0.939$).
2. **Operational Tipping Point Intervention**:
   - Automated Trigger: When any subscriber reaches **$3$ customer service calls**, immediately flag their account and route their next call to a senior retention specialist.
3. **Bill Shock Proactive Outreach**:
   - Automatically notify customers whose `total_bill` crosses **$\$75$** or whose daytime usage exceeds $250$ minutes, offering a tailored flat-rate plan before they switch.
4. **Plan Cross-Selling**:
   - Actively bundle `voice_mail_plan` with international plans to reduce international churn by $\sim 10\%$.

---

## 12. Key Lessons Learned
- **Aesthetics & Question Evolution**: Univariate $\to$ Bivariate $\to$ Multivariate progression uncovered the complete non-linear story before any modeling began.
- **Visuals must match Data Types**: Discrete categories need contingency tables and proportional bars, not continuous KDE contours.
- **Linear Model Limits**: Linear models hit a hard ceiling ($F1 \sim 0.628$) because real-world customer churn is governed by sharp step-functions ($\ge 4$ calls) and bimodal bill distributions.
- **Tree Ensembles Rule Tabular Data**: Decision Trees and Gradient Boosters naturally shattered the linear ceiling, reaching **$0.935$ F1**.
