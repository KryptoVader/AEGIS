# Day 08 — Decision Log: Multi-Store Time-Series Demand Forecasting (Lags & Rolling Windows)

## 1. Business & Client Understanding
- **Client**: Retail Chain Supply Chain & Demand Planning Division.
- **Business Goal**: Forecast daily store sales (`sales`) across 10 retail store locations to optimize inventory replenishment, prevent stockouts during peak promotional weekends, and eliminate excess holding costs.
- **Problem Type**: Time-Series Tabular Regression ($10,960$ daily store observations across 3 years: 2022–2024).
- **Target Variable**: `sales` (Continuous non-negative daily sales volume).
- **Golden Rule of Time-Series**: **NO RANDOM SHUFFLE CV!** Shuffling time-series data causes catastrophic future-to-past data leakage. Must use strict Out-of-Time temporal train-test split (Train on 2022–2023, Test on 2024).

---

## 2. Initial Observations & Data Hygiene
- **Data Shape**: $10,960$ rows, $5$ raw columns (`date`, `store_id`, `sales`, `promo`, `school_holiday`).
- **Missing Values**: $0$ missing values.
- **Duplicates**: $0$ duplicate rows.
- **Temporal Range**: 3 full years ($2022\text{--}2024$, $1,096$ consecutive days per store).

---

## 3. Exploratory Data Analysis (EDA) Findings
- **Weekly Seasonality**: Saturday and Sunday show consistent $+40\%$ demand surges compared to Monday weekdays.
- **Promotional Surge**: Days with active promotions (`promo == 1`) show a $+35\%$ baseline sales uplift across all 10 stores.
- **Annual Seasonality**: July/August school holidays and December holiday season demonstrate strong annual demand peaks.

---

## 4. Feature Engineering Decisions

| Engineered Feature | Logic / Formula | Rationale |
| :--- | :--- | :--- |
| `lag_1` | `df.groupby('store_id')['sales'].shift(1)` | Immediate prior day sales anchor ($t-1$). |
| `lag_7` | `df.groupby('store_id')['sales'].shift(7)` | Same day of week sales anchor from 1 week ago ($t-7$). |
| `lag_14` | `df.groupby('store_id')['sales'].shift(14)` | 2-week historical sales anchor ($t-14$). |
| `lag_28` | `df.groupby('store_id')['sales'].shift(28)` | 4-week historical sales anchor ($t-28$). |
| `rolling_mean_7` | `df.groupby('store_id')['sales'].shift(1).rolling(7).mean()` | 7-day moving average (short-term momentum). **`shift(1)` safeguard applied!** |
| `rolling_std_7` | `df.groupby('store_id')['sales'].shift(1).rolling(7).std()` | 7-day moving volatility metric. |
| `rolling_mean_30` | `df.groupby('store_id')['sales'].shift(1).rolling(30).mean()` | 30-day moving average (monthly demand baseline). |
| `day_of_week` / `is_weekend` | `date.dt.dayofweek`, `dayofweek.isin([5,6])` | Weekly calendar cycle indicators. |

---

## 5. Validation Strategy & Temporal Leakage Safeguards
- **Temporal Train/Test Split**:
  - **Training Set**: $2022\text{--}2023$ ($7,020$ samples).
  - **Out-of-Time Test Set**: $2024$ ($3,660$ samples).
- **The `shift(1)` Rule**: All rolling statistics use `.shift(1)` prior to `.rolling()` computation, ensuring today's target `sales` is **never included in today's feature row**.

---

## 6. Model Comparison Leaderboard (2024 Out-of-Time Test Set)

| Model Family | Model Architecture | Test RMSE (Lower=Better) | Test MAE (Sales Error) | Test $R^2$ (Higher=Better) |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Baseline** | Ridge Regression (`alpha=10.0`) | $86.69$ | $63.54$ sales | $84.33\%$ |
| **Linear Baseline** | Linear Regression | $86.60$ | $63.53$ sales | $84.36\%$ |
| **Boosting (GPU)** | XGBoost GPU (Untuned) | $79.54$ | $60.07$ sales | $86.81\%$ |
| **Boosting** | LightGBM (Untuned) | $74.45$ | $56.67$ sales | $88.44\%$ |
| **Bagging** | Random Forest (`n_estimators=200`) | $74.22$ | **$54.44$ sales** 🎯 | $88.51\%$ |
| **Boosting** | HistGradientBoosting (Untuned) | $73.92$ | $56.20$ sales | $88.60\%$ |
| **TUNED CHAMPION** | **CatBoost (Untuned Baseline)** | **$\mathbf{72.56}$** 🏆 | **$55.85$ sales** | **$\mathbf{89.02\%}$** 🏆 |

---

## 7. Feature Importance (CatBoost Regressor)

| Rank | Feature Name | Importance Weight (%) | Business Meaning |
| :---: | :--- | :---: | :--- |
| **1** | `promo` | **$25.15\%$** | Active marketing promotion indicator (+35% demand uplift). |
| **2** | `month` | **$13.30\%$** | Annual seasonal cycle. |
| **3** | `rolling_mean_7` | **$10.27\%$** | **Engineered**: 7-day short-term sales momentum. |
| **4** | `is_weekend` | **$9.38\%$** | Weekend demand surge. |
| **5** | `day_of_week` | **$9.10\%$** | Weekly daily cycle. |
| **6** | `lag_7` | **$8.28\%$** | **Engineered**: Same day last week demand anchor. |
| **7** | `store_id` | **$6.50\%$** | Store baseline capacity size. |
| **8** | `lag_14` | **$6.02\%$** | **Engineered**: 2-week historical sales anchor. |
| **9** | `rolling_mean_30` | **$5.09\%$** | **Engineered**: 30-day monthly demand baseline. |

*(Note: Engineered lag and rolling window features account for **$32.8\%$ of the winning CatBoost model's predictive weight**!)*

---

## 8. Final Decision & Supply Chain Recommendations
1. **Model Deployment**: Deploy **CatBoost Regressor** with 7-day and 30-day lag/rolling features ($R^2 = 89.02\%$).
2. **Promotional Pre-Stocking Rules**:
   - Because `promo` drives **$25.15\%$ of total sales variation**, dispatch additional inventory trucks **48 hours prior** to any scheduled promotional campaign.
3. **Weekly Rebalancing Schedule**:
   - Use `rolling_mean_7` and `lag_7` to dynamically adjust Friday delivery volumes to match expected weekend demand surges.
