# Day 03 — Decision Log: Bank Direct Marketing & Term Deposit Prediction

## 1. Business & Client Understanding
- **Client**: Retail Banking Direct Marketing & Growth Division.
- **Business Goal**: Predict which prospective bank clients will subscribe to a long-term term deposit (`y = 1`) when contacted via direct phone marketing campaigns, allowing the bank to prioritize high-probability leads and maximize campaign ROI while minimizing call center labor expenses and customer fatigue.
- **Operational Reality**:
  - *Base Conversion Rate*: Only $\approx 11.26\%$ of cold-called leads convert ($4,640$ subscribers out of $41,188$ total contacts).
  - *Call Center Cost*: Every phone call incurs agent labor cost ($\sim \$5\text{–}\$10/\text{call}$).
  - *Product Margin*: A new term deposit account generates $\sim \$200\text{–}\$500$ in lifetime net interest margin.
  - *Target*: `y` ($1$ = Subscribed, $0$ = Did not subscribe).

---

## 2. Initial Observations & Data Hygiene
- **Data Shape**: $41,188$ rows, $21$ columns.
- **Missing Values**: $0$ standard `NaN` values.
- **Duplicates**: $12$ duplicate rows detected and **dropped** ($41,176$ clean rows remaining).
- **Target Distribution**: $88.73\%$ Non-subscribers ($36,548$) vs $11.27\%$ Subscribers ($4,640$).

---

## 3. The Critical ML & Domain Finding: Target Leakage in `duration`
- **The Finding**: Call duration (`duration` in seconds) has extreme correlation with the target ($20$-minute calls almost always convert, while $15$-second calls do not).
- **The Operational Leakage Trap**: Call duration is strictly an *after-the-fact* measurement. In production, at 9:00 AM before an agent dials the phone, `duration = 0` for all prospective leads!
- **Data Science Decision**: **DROP `duration` completely from the feature matrix $X$**. This guarantees a genuine, realistic lead scoring engine that operates on prospective client profile, past history, and macroeconomic timing.

---

## 4. Exploratory Data Analysis (EDA) Findings

### Pillar 1: Sentinel Value Handling (`pdays`)
- `pdays == 999` is an imputed sentinel code representing *"never contacted in a prior campaign"*.
- Only $\sim 3.8\%$ of clients have an actual day count ($0\text{–}27$ days).
- **Action**: Engineer `previously_contacted = (pdays != 999).astype(int)` and clean `pdays` by replacing $999$ with $-1$.

### Pillar 2: Macroeconomic Environment (`euribor3m`, `emp_var_rate`, `nr_employed`)
- `euribor3m` shows a distinct bimodal distribution:
  - *Low Rate Regime ($1.0\text{–}1.5\%$)*: High subscriber conversion mass (bank fixed deposits appear highly lucrative).
  - *High Rate Regime ($4.5\text{–}5.0\%$)*: Very low conversion (market rates compete with bank deposits).

### Pillar 3: Contact Fatigue (`campaign`)
- Conversions are heavily concentrated within $1\text{–}3$ phone calls (max $4$).
- Beyond $6$ calls, conversion drops to essentially $0\%$.
- **Action**: Calling a client more than $4$ times is operational waste.

### Pillar 4: Demographics & Prior Success
- `poutcome == 'success'`: Clients who subscribed to a previous campaign have a massive **$\sim 65\%$ repeat subscription rate**.
- `job`: Students and Retirees have the highest conversion rates ($\sim 25\text{–}31\%$), while blue-collar and service workers convert at only $\sim 6\text{–}8\%$.

---

## 5. Feature Engineering Decisions

| Engineered Feature | Logic / Formula | Rationale |
| :--- | :--- | :--- |
| `previously_contacted` | `(df['pdays'] != 999).astype(int)` | Binary flag isolating previously engaged customers from cold leads. |
| `pdays_clean` | `df['pdays'].replace(999, -1)` | Prevents models from treating 999 as a literal quantitative 999-day duration. |
| `contact_fatigue` | `(df['campaign'] >= 4).astype(int)` | Flags diminishing-return outreach threshold. |
| `is_student_or_retired` | `df['job'].isin(['student', 'retired']).astype(int)` | Explicit indicator for high-propensity saving demographic groups. |

---

## 6. Preprocessing & Leakage Prevention Pipeline
- **`ColumnTransformer`**:
  - `StandardScaler`: Fitted on continuous macroeconomic indicators and quantitative counts.
  - `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')`: Fitted on nominal categories (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`).
  - `passthrough`: Binary flags (`previously_contacted`, `contact_fatigue`, `is_student_or_retired`).
- **Validation**: Strict 5-Fold `StratifiedKFold(shuffle=True, random_state=42)` across all evaluations.

---

## 7. Model Comparison Leaderboard (5-Fold Stratified CV)

| Model Family | Model Architecture | 5-Fold Mean ROC-AUC | 5-Fold Mean F1 | 5-Fold Mean Precision | 5-Fold Mean Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Linear** | Ridge Classifier (`class_weight='balanced'`) | $0.7906$ | $0.4468$ | $34.67\%$ | $62.88\%$ |
| **Linear Baseline** | Logistic Regression (`class_weight='balanced'`) | $0.7913$ | $0.4506$ | $35.17\%$ | $62.73\%$ |
| **Single Tree** | Decision Tree (`max_depth=8`) | $0.7719$ | $0.4509$ | $36.08\%$ | $60.21\%$ |
| **Boosting (GPU)** | XGBoost GPU (`scale_pos_weight=7.87`) | $0.7766$ | $0.4475$ | $36.25\%$ | $58.48\%$ |
| **Boosting** | CatBoost (`auto_class_weights='Balanced'`) | $0.7923$ | $0.4657$ | $37.30\%$ | $62.02\%$ |
| **Boosting** | LightGBM (`class_weight='balanced'`) | $0.8011$ | $0.4751$ | $38.07\%$ | $63.20\%$ |
| **Bagging** | Random Forest (`n_estimators=200, max_depth=12`) | $0.8013$ | $0.4874$ | $40.53\%$ | $61.13\%$ |
| **Boosting** | HistGradientBoosting (Untuned) | $0.8029$ | $0.4762$ | $38.22\%$ | $63.20\%$ |
| **TUNED CHAMPION** | **HistGradientBoosting (Tuned + Threshold $\tau^* = 0.23$)** | **$\mathbf{0.8062}$** | **$\mathbf{0.5075}$** | **$45.95\%$** | **$56.67\%$** |
| **FEATURE SELECTED** | **HistGradientBoosting + SelectKBest ($k=25$)** | **$\mathbf{0.8068}$** 🏆 | $0.3668$ | **$\mathbf{66.25\%}$** 🎯 | $25.37\%$ |

---

## 8. Permutation Feature Importance & Drivers

| Rank | Feature Name | Permutation Importance | Business Meaning |
| :---: | :--- | :---: | :--- |
| **1** | `nr_employed` | **$0.0144$** | Macroeconomic employment level (economic expansion vs recession). |
| **2** | `contact` | **$0.0033$** | Communication channel (Cellular vs Landline). |
| **3** | `emp_var_rate` | **$0.0027$** | Employment variation rate (economic shift indicator). |
| **4** | `pdays_clean` | **$0.0025$** | Recency of past campaign interaction. |
| **5** | `month` | **$0.0024$** | Seasonal campaign timing (March/September/October spikes). |
| **6** | `poutcome` | **$0.0023$** | Past campaign outcome (Success vs Failure). |
| **7** | `euribor3m` | **$0.0021$** | 3-month Euribor benchmark interest rate. |

---

## 9. Final Business Strategy & Operational Recommendations
1. **Model Deployment**: Deploy the tuned **HistGradientBoosting + Feature Selection Pipeline** ($ROC\text{-}AUC = 0.807$).
2. **Tripling Agent Efficiency**:
   - Random cold outreach converts at only **$11.3\%$**.
   - Operating at the selected model threshold achieves **$46\%\text{–}66\%$ Precision**, tripling to quadrupling deposit sales per agent-hour.
3. **Hard Cap on Call Retries**:
   - Establish an operational policy capping outbound calls at **$3$ contacts maximum** per client per quarter to eliminate wasted call labor and customer annoyance.
4. **Macroeconomic Market Timing**:
   - Launch high-intensity term deposit telemarketing campaigns during **low-Euribor periods ($< 2.0\%$)**, when bank fixed-deposit yield spreads are most attractive to retail savers.
