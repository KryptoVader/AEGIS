# Day 04 — Decision Log: Hotel Booking Demand & Revenue Management

## 1. Business & Client Understanding
- **Client**: Revenue Management & Operations Division of a Hospitality Chain (Resort & City Hotels).
- **Business Goal**: Predict whether a hotel booking will be canceled (`is_canceled = 1`) prior to arrival, allowing the revenue management team to implement intelligent overbooking strategies, dynamically adjust cancellation policies, and prevent revenue loss from empty rooms.
- **Cost/Impact Trade-off**:
  - *False Positive (Predicting cancellation for a guest who actually shows up)*: Overbooking risk — if no room is available, the hotel must "walk" the guest to a competitor hotel, paying for their accommodation, transportation, and suffering reputational damage.
  - *False Negative (Failing to predict a cancellation)*: Opportunity cost — the room sits empty, generating zero revenue ($ADR$).
- **Target Variable**: `is_canceled` ($1$ = Canceled, $0$ = Not Canceled / Checked-in).
- **Dataset Scale**: $119,390$ bookings, $32$ features.

## 2. Initial Observations & Data Hygiene

## 3. Hypotheses & Domain Inquiries

## 4. Exploratory Data Analysis (EDA) Findings

## 5. Data Quality & Cleaning Decisions

## 6. Feature Engineering Decisions

## 7. Preprocessing & Leakage Prevention Decisions

## 8. Validation Strategy & Metric Selection

## 9. Baseline Model

## 10. Model Selection & Exploration

## 11. Hyperparameter Tuning & Threshold Optimization

## 12. Error Analysis & Interpretability

## 13. Final Decision & Business Recommendation

## 14. What Worked vs. What Failed

## 15. Key Lessons Learned
