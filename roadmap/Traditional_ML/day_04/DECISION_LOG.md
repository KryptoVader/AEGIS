# Day 04 — Decision Log: Hotel Booking Demand & Revenue Management

## 1. Business & Client Understanding
- **Client**: Revenue Management & Operations Division of a Hospitality Chain (Resort & City Hotels).
- **Business Goal**: Predict whether a hotel booking will be canceled (`is_canceled = 1`) prior to arrival, allowing the revenue management team to implement intelligent overbooking strategies, dynamically adjust cancellation policies, and prevent revenue loss from empty rooms.
- **Cost/Impact Trade-off**:
  - *False Positive (Overbooking Risk / Walking a Guest)*: Predicting a cancellation for a guest who actually shows up. If no rooms remain, the hotel must "walk" the guest to a competitor hotel, paying for their stay, transportation, and suffering brand reputational damage.
  - *False Negative (Opportunity Cost / Empty Room)*: Failing to predict a cancellation leaves the perishable room asset vacant, generating zero revenue ($ADR$).
- **Target Variable**: `is_canceled` ($1$ = Canceled, $0$ = Checked-in).
- **Dataset Scale**: $119,390$ bookings, $32$ features.

---

## 2. Initial Observations & Data Quality
- **Data Shape**: $119,390$ rows, $32$ columns.
- **Missing Values**:
  - `company`: $112,593$ missing ($94.3\%$) $\to$ Engineer `has_company = df['company'].notna().astype(int)`.
  - `agent`: $16,340$ missing ($13.7\%$) $\to$ Engineer `has_agent = df['agent'].notna().astype(int)`.
  - `country`: $488$ missing ($0.41\%$) $\to$ Impute with `'Unknown'`.
  - `children`: $4$ missing ($0.003\%$) $\to$ Impute with $0$.
- **Invalid Records Removal**: Dropped $181$ bad records with zero total guests (`adults + children + babies == 0`) or negative room rate (`adr < 0`). $119,209$ clean records remaining.

---

## 3. The Critical Target Leakage Discovery
- **Target Leakage Columns**: `reservation_status` (contains `'Check-Out'`, `'Canceled'`, `'No-Show'`) and `reservation_status_date`.
- **Why it is Leakage**: `reservation_status` is updated by the front desk only *after* the guest checks in or cancels. Including it in model training yields $100\%$ accuracy artificially, but fails in production prior to arrival date.
- **Data Science Decision**: **DROP `reservation_status` and `reservation_status_date`** from the feature matrix $X$.

---

## 4. Exploratory Data Analysis (EDA) Findings (The 5 Pillars)

### Pillar 1: Temporal Lead Time & Seasonality
- `lead_time` follows a right-skewed Gamma/Log-Normal distribution ($0\text{–}700+$ days).
- Bookings with short lead times ($< 30$ days) rarely cancel.
- Bookings made $> 150\text{–}200+$ days in advance suffer high cancellation rates due to long-term planning uncertainty.

### Pillar 2: Party Composition & Demographics
- `country`: Domestic travelers (`PRT` - Portugal) show distinctly higher cancellation rates compared to international inbound travelers.

### Pillar 3: Financial & Commercial Mechanics (Simpson's Paradox)
- `deposit_type == 'Non Refund'` has a **$> 99\%$ cancellation rate**!
- **Multivariate Interaction Discovery**: $97.2\%$ of all `Non Refund` bookings ($14,178 / 14,587$) originate from `Groups` and `Offline TA/TO` (Travel Agents / Tour Operators) who book massive bulk blocks 6–12 months in advance and cancel entire 100-room blocks at once.

### Pillar 4: Room Dynamics & Behavioral Friction
- `room_mismatch` (`reserved_room_type != assigned_room_type`): Guests who receive a room reassignment or complimentary upgrade cancel **only $5.4\%$ of the time** (compared to $41.6\%$ for non-reassigned guests).

### Pillar 5: Guest History & Loyalty Anchor
- `previous_cancellations >= 1`: Serial cancelers repeat their behavior ($> 90\%$ repeat cancellation rate!).

---

## 5. Feature Engineering Decisions

| Engineered Feature | Logic / Formula | Rationale |
| :--- | :--- | :--- |
| `room_mismatch` | `(reserved_room_type != assigned_room_type).astype(int)` | Room upgrade / reassignment anchor ($5.4\%$ cancellation rate). |
| `is_group_non_refund` | `((market_segment == 'Groups') & (deposit_type == 'Non Refund')).astype(int)` | Captures the single largest bulk cancellation hazard interaction. |
| `cancel_ratio` | `previous_cancellations / (previous_cancellations + previous_bookings_not_canceled + 1)` | Quantitative measure of historical guest cancellation risk. |
| `total_guests` | `adults + children + babies` | Total party size. |
| `is_family` | `((children > 0) \| (babies > 0)).astype(int)` | Family booking indicator. |
| `total_stay_nights` | `stays_in_weekend_nights + stays_in_week_nights` | Total stay duration. |
| `has_previous_cancellations` | `(previous_cancellations > 0).astype(int)` | Binary serial canceler flag. |
| `has_company` / `has_agent` | `df['company'].notna().astype(int)` | Binarized corporate and travel agency account indicators. |

---

## 6. Preprocessing & Leakage Prevention Pipeline
- **`ColumnTransformer`**:
  - `Pipeline(SimpleImputer(strategy='median') -> StandardScaler)` on continuous/count numerical features.
  - `Pipeline(SimpleImputer(strategy='constant', fill_value='Unknown') -> OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))` on categorical columns.
  - `passthrough` on binary engineered flags.
- **Validation Scheme**: 5-Fold `StratifiedKFold(shuffle=True, random_state=42)` across $119,209$ clean records.

---

## 7. Model Comparison Leaderboard (5-Fold Stratified CV)

| Model Family | Model Architecture | 5-Fold Mean ROC-AUC | 5-Fold Mean F1 | 5-Fold Mean Precision | 5-Fold Mean Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Linear** | Ridge Classifier (`class_weight='balanced'`) | $0.8961$ | $0.7543$ | $72.76\%$ | $78.30\%$ |
| **Linear Baseline** | Logistic Regression (`class_weight='balanced'`) | $0.9013$ | $0.7583$ | $73.04\%$ | $78.83\%$ |
| **Single Tree** | Decision Tree (`max_depth=10`) | $0.9169$ | $0.7792$ | $76.20\%$ | $79.73\%$ |
| **Bagging** | Random Forest (`n_estimators=200, max_depth=15`) | $0.9348$ | $0.8110$ | **$80.27\%$** | $81.94\%$ |
| **Boosting** | HistGradientBoosting (Untuned) | $0.9427$ | $0.8177$ | $78.36\%$ | $85.49\%$ |
| **Boosting** | LightGBM (`class_weight='balanced'`) | $0.9429$ | $0.8191$ | $78.45\%$ | $85.71\%$ |
| **Boosting** | CatBoost (Untuned) | $0.9473$ | $0.8280$ | $79.47\%$ | $86.41\%$ |
| **Boosting (GPU)** | XGBoost GPU (Untuned) | $0.9460$ | $0.8253$ | $79.07\%$ | $86.31\%$ |
| **TUNED CHAMPION** | **GPU XGBoost (Tuned + Threshold $\tau^* = 0.51$)** | **$\mathbf{0.9489}$** 🏆 | **$\mathbf{0.8311}$** 🏆 | **$81.66\%$** | **$84.60\%$** |

---

## 8. Feature Importance (GPU XGBoost Pipeline)

| Rank | Feature | Importance Weight | Business Meaning & EDA Verification |
| :---: | :--- | :---: | :--- |
| **1** | `is_group_non_refund` | **$0.3646$** ($36.5\%$) | **Engineered Feature #1**: Tour group bulk non-refundable contract risk. |
| **2** | `deposit_type_Non Refund` | **$0.2687$** ($26.9\%$) | Non-refundable deposit indicator (combined with #1 accounts for $63.4\%$ of model). |
| **3** | `required_car_parking_spaces` | **$0.0446$** ($4.5\%$) | Parking commitment signal (guests requiring parking rarely cancel). |
| **4** | `room_mismatch` | **$0.0375$** ($3.8\%$) | **Engineered Feature #2**: Room upgrade / reassignment anchor ($5.4\%$ cancel rate). |
| **5** | `market_segment_Online TA` | **$0.0266$** ($2.7\%$) | Online Travel Agency channel. |
| **6** | `country_PRT` | **$0.0229$** ($2.3\%$) | Domestic (Portugal) guest indicator. |
| **7** | `cancel_ratio` | **$0.0139$** ($1.4\%$) | **Engineered Feature #3**: Quantitative serial cancellation history. |

---

## 9. Final Decision & Revenue Management Recommendations
1. **Deployment**: Deploy the tuned **GPU XGBoost Pipeline** ($ROC\text{-}AUC = 0.9489, F1 = 0.8311$).
2. **Intelligent Overbooking Algorithm**:
   - For Tour Group `Non Refund` bookings, expect a $\sim 99\%$ cancellation rate and allow overbooking for those specific room blocks.
3. **Commitment Signals**:
   - Guests requesting parking spaces or receiving room upgrades should be marked as **Guaranteed Check-Ins ($95\%+$ fulfillment)**.
4. **Lead Time Overbooking Curve**:
   - Implement dynamic overbooking limits that scale upwards for bookings made $> 150$ days in advance.
