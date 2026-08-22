# Day 09 — Decision Log: Extreme Financial Fraud & Anomaly Detection

## 1. Business & Client Understanding
- **Client**: Financial Crime & Fraud Prevention Division of a Global Payment Processor.
- **Business Goal**: Detect illegal fraudulent credit card transactions (`Class = 1`) in real-time streams to block compromised cards before financial authorization, while minimizing false alarms on legitimate cardholders (`Class = 0`).
- **Extreme Asymmetric Cost Matrix**:
  - *False Negative (Missed Fraud)*: A $\$2,500$ fraudulent charge passes through; the bank incurs chargeback loss, VISA/Mastercard fines, and fraud investigation labor.
  - *False Positive (False Alarm)*: A cardholder's legitimate transaction at a grocery store is declined; customer friction, embarrassing decline at POS, and customer churn.
- **Target Variable**: `Class` ($1$ = Fraud, $0$ = Legitimate).
- **Extreme Imbalance Scale**: $284,807$ transactions, $30$ features (`Time`, PCA features `V1`..`V28`, `Amount`).
  - Legitimate ($0$): $284,315$ ($99.828\%$)
  - Fraudulent ($1$): $492$ (**$0.172\%$** — 1 fraud in every 578 transactions!).

---

## 2. Initial Observations & Data Hygiene

## 3. Hypotheses & Domain Inquiries

## 4. Exploratory Data Analysis (EDA) Findings

## 5. Data Quality & Cleaning Decisions

## 6. Advanced Anomaly & Fraud Feature Engineering

## 7. Preprocessing & Validation Decisions (PR-AUC vs ROC-AUC)

## 8. Validation Strategy (Stratified PR-AUC Cross-Validation)

## 9. Baseline Model

## 10. Model Selection (Isolation Forest vs Cost-Sensitive Boosting)

## 11. Precision-Recall Threshold Optimization

## 12. Key Lessons Learned
