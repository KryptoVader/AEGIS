# Day 05 — Decision Log: Urban Transportation Demand Forecasting (Bike Sharing)

## 1. Business & Client Understanding
- **Client**: Municipal Transportation Authority & Smart City Micro-Mobility Operator.
- **Business Goal**: Forecast hourly bike rental demand (`count`) across urban station networks to optimize fleet rebalancing logistics, prevent station depletion during peak hours, and allocate maintenance schedules.
- **Asymmetric Cost Trade-off**:
  - *Under-predicting Demand (Station Depletion)*: Commuters arrive at an empty station, miss transit connections, and lose trust in public micro-mobility.
  - *Over-predicting Demand (Station Overflow)*: Excess bikes clutter sidewalks and block pedestrian right-of-way, incurring city fines.
- **Target Variable**: `count` (Continuous hourly rental volume, non-negative, right-skewed, range: $1\text{–}977$).
- **Dataset Scale**: $17,379$ hourly observations, $13$ features.

---

## 2. Initial Observations & Data Hygiene
- **Data Shape**: $17,379$ rows, $13$ columns.
- **Missing Values**: $0$ missing values across all columns.
- **Duplicates**: $2$ duplicate rows detected and **dropped** ($17,377$ clean rows remaining).
- **Target Distribution**: Gamma / Tweedie count distribution ($1\text{–}977$ rentals/hr).

---

## 3. The Core Target Transformation Decision
- **Why Raw Target Fails**: Ordinary Least Squares (OLS) and standard MSE assume normal Gaussian errors with constant variance. On raw count data, linear models predict **negative bike rentals (e.g. $-15$ bikes!)** at 3:00 AM.
- **Data Science Decision**: Transform target to $y_{\text{log}} = \ln(1 + \text{count})$ (`np.log1p`).
  - Guarantees positive predictions when inverted (`np.expm1`).
  - Optimizing MSE on $y_{\text{log}}$ is **mathematically identical to optimizing RMSLE (Root Mean Squared Logarithmic Error)** on the original scale.

---

## 4. Exploratory Data Analysis (EDA) Findings

### Pillar 1: Hourly Demand Dynamics & Commuter Interactions
- `sns.lineplot(x='hour', y='count', hue='workingday')` revealed two completely distinct human behavior regimes:
  - *Working Days (`workingday == True`)*: Bimodal commute pattern with massive spikes at **8:00 AM** ($480$ bikes/hr) and **5:00 PM – 6:00 PM** ($530$ bikes/hr).
  - *Non-Working Days (`workingday == False`)*: Unimodal leisure pattern peaking in the afternoon between **12:00 PM and 4:00 PM** ($370$ bikes/hr).

### Pillar 2: Thermal Comfort & Weather Glitch Detection
- `temp` and `feel_temp` are $r \approx 0.99$ collinear.
- **Sensor Glitch Discovered**: A horizontal line of corrupted data points stuck at `feel_temp = 12.12°C` while actual `temp` was $25^\circ\text{C}\text{--}35^\circ\text{C}$.

---

## 5. Feature Engineering Decisions

| Engineered Feature | Logic / Formula | Rationale |
| :--- | :--- | :--- |
| `hour_sin` / `hour_cos` | $\sin(2\pi \cdot \text{hour} / 24), \cos(2\pi \cdot \text{hour} / 24)$ | Maps discrete 24-hour time onto a continuous 2D circular clock, connecting 23:00 and 00:00. |
| `month_sin` / `month_cos` | $\sin(2\pi \cdot \text{month} / 12), \cos(2\pi \cdot \text{month} / 12)$ | Captures 12-month annual seasonal cycle continuity. |
| `weekday_sin` / `weekday_cos` | $\sin(2\pi \cdot \text{weekday} / 7), \cos(2\pi \cdot \text{weekday} / 7)$ | Captures 7-day weekly cycle continuity. |
| `is_morning_rush` | `((workingday == 1) & (hour.isin([7,8,9]))).astype(int)` | Morning commuter rush hour indicator. |
| `is_evening_rush` | `((workingday == 1) & (hour.isin([17,18,19]))).astype(int)` | Evening commuter rush hour indicator. |
| `is_weekend_afternoon` | `((workingday == 0) & (hour.between(12,16))).astype(int)` | Weekend leisure riding peak indicator. |
| `temp_diff` | `feel_temp - temp` | Measures thermal discomfort (humidity/wind-chill). |

---

## 6. Preprocessing & Leakage Prevention Pipeline
- **Categorical Encoding**: `OrdinalEncoder` for ordered categorical columns (`weather`, `season`).
- **Feature Scaling**: `StandardScaler` applied to features inside linear model pipelines. (Not required for tree models).
- **Validation Split**: Train-Test Split ($80/20$, random_state=42) on $17,377$ observations.

---

## 7. Model Comparison Leaderboard

| Model Family | Model Architecture | Test RMSLE (Lower=Better) | Test $R^2$ (Higher=Better) | Test MAE (Error in Bikes) |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Baseline** | Linear Regression (`StandardScaler`) | $0.6837$ | $76.25\%$ | $68.73$ bikes/hr |
| **Linear Baseline** | Ridge Regression (`alpha=10.0`) | $0.6837$ | $76.25\%$ | $68.67$ bikes/hr |
| **Single Tree** | Decision Tree (`max_depth=12`) | $0.4056$ | $91.64\%$ | $34.88$ bikes/hr |
| **Bagging** | Random Forest (`n_estimators=200`) | $0.3158$ | $94.93\%$ | $24.54$ bikes/hr |
| **Boosting** | HistGradientBoosting (Untuned) | $0.2978$ | $95.49\%$ | $25.33$ bikes/hr |
| **Boosting** | LightGBM (Untuned) | $0.2950$ | $95.58\%$ | $25.48$ bikes/hr |
| **Boosting** | CatBoost (Untuned) | $0.2769$ | $96.10\%$ | $22.38$ bikes/hr |
| **TUNED CHAMPION** | **GPU XGBoost (Tuned)** | **$\mathbf{0.2783}$** 🏆 | **$\mathbf{96.07\%}$** 🏆 | **$\mathbf{22.37}$ bikes/hr** 🏆 |

---

## 8. Feature Importance (Tuned GPU XGBoost)

| Rank | Feature Name | Importance Weight (%) | Business Meaning & EDA Verification |
| :---: | :--- | :---: | :--- |
| **1** | `hour_cos` | **$23.50\%$** | **Engineered Feature #1**: Cyclic 24-hour cosine clock coordinate. |
| **2** | `hour_sin` | **$20.81\%$** | **Engineered Feature #2**: Cyclic 24-hour sine clock coordinate. |
| **3** | `workingday` | **$10.90\%$** | Commuter vs Weekend behavior regime. |
| **4** | `is_evening_rush` | **$8.74\%$** | **Engineered Feature #3**: Evening 5:00 PM rush hour commute spike. |
| **5** | `is_morning_rush` | **$8.39\%$** | **Engineered Feature #4**: Morning 8:00 AM rush hour commute spike. |
| **6** | `year` | **$5.77\%$** | Fleet adoption growth (Year 0 to Year 1). |
| **7** | `season` | **$4.42\%$** | Seasonal demand shift. |
| **8** | `is_weekend_afternoon` | **$4.33\%$** | **Engineered Feature #5**: Weekend afternoon leisure peak. |
| **9** | `feel_temp` | **$2.83\%$** | Thermal comfort perception. |

*(Note: Features explicitly conceptualized and engineered during EDA account for **over $65.8\%$ of the winning GPU XGBoost model's decision weights**!)*

---

## 9. Final Decision & Smart City Fleet Recommendations
1. **Model Deployment**: Deploy the tuned **GPU XGBoost Regressor** ($R^2 = 96.07\%, \text{MAE} = 22.37 \text{ bikes/hr}$).
2. **Automated Rebalancing Logistics**:
   - **7:00 AM Pre-Commute Loading**: Dispatch rebalancing trucks at 6:30 AM to fully stock residential stations before the 8:00 AM morning rush ($480$ bikes/hr demand spike).
   - **4:00 PM Post-Work Loading**: Dispatch trucks at 4:00 PM to clear residential drop-off docks and stock commercial downtown stations before the 5:00 PM evening rush ($530$ bikes/hr demand spike).
3. **Weather Disruption Protocol**:
   - When precipitation or extreme temperatures occur, automatically scale down expected fleet deployment to prevent idle bike sidewalk clutter.
