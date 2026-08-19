# Day 06 — Decision Log: Multi-Class Wine Quality Grading

## 1. Business & Client Understanding
- **Client**: Sommelier Operations & Quality Control Division of an International Winery.
- **Business Goal**: Automate chemical quality grading of wine batches prior to bottling, categorizing wines into discrete quality rating scores ($3, 4, 5, 6, 7, 8$), to optimize pricing tiers and prevent defective batches from reaching consumers.
- **Problem Type**: Multi-Class Imbalanced Classification ($6$ discrete rating classes).
- **Target Variable**: `quality` ($3, 4, 5, 6, 7, 8$).
- **Dataset Scale**: $1,599$ samples, $11$ physicochemical features.

---

## 2. Initial Observations & Data Hygiene
- **Data Shape**: $1,599$ rows, $12$ columns.
- **Missing Values**: $0$ missing values.
- **Duplicates**: $240$ duplicate rows detected.
- **Target Distribution**: Extreme Multi-Class Imbalance:
  - Class $3$: $10$ samples ($0.6\%$)
  - Class $4$: $53$ samples ($3.3\%$)
  - Class $5$: $577$ samples ($36.1\%$)
  - Class $6$: $535$ samples ($33.5\%$)
  - Class $7$: $167$ samples ($10.4\%$)
  - Class $8$: $17$ samples ($1.1\%$)

---

## 3. Exploratory Data Analysis (EDA) Findings
- **`alcohol` ($r = +0.48$)**: #1 positive correlation with high quality ratings. Higher alcohol volume marks premium vintages.
- **`volatile acidity` ($r = -0.39$)**: #1 negative correlation. High volatile acidity produces acetic acid (vinegar taste), destroying wine quality.
- **`sulphates` ($r = +0.25$) & `citric acid` ($r = +0.23$)**: Positive contributors to freshness and flavor preservation.

---

## 4. Why 6-Class Macro F1 is Low ($\sim 0.31$) — The 3 Root Causes

1. **Extreme Rare Class Starvation (Classes 3 & 8)**:
   - Class $3$ has only $10$ total samples; Class $8$ has only $17$ total samples.
   - In 5-Fold Stratified CV, each validation fold has only **$2$ samples of Class 3** and **$3$ samples of Class 8**.
   - When a model gets $0$ recall on Class 3, $F_{1,\text{Class 3}} = 0.0$. Because Macro F1 averages across all 6 classes equally:
     $$\text{Macro F1} = \frac{F_{1,c3} + F_{1,c4} + F_{1,c5} + F_{1,c6} + F_{1,c7} + F_{1,c8}}{6}$$
     $0.0$ scores on rare classes $3$ and $8$ drag down the overall unweighted Macro F1.
2. **Subjective Human Sommelier Noise**:
   - Human tasting panels grade wines on a subjective continuous spectrum. The chemical difference between a "5" and a "6" wine is subtle, causing confusion matrix errors to cluster almost entirely on adjacent classes ($5 \leftrightarrow 6$).
3. **Continuous-to-Discrete Binning Discrepancy**:
   - Quality ratings are naturally ordinal/continuous rather than completely independent nominal classes.

---

## 5. Preprocessing & Validation Strategy
- **StandardScaler**: Applied to all 11 physicochemical features for linear models.
- **Validation**: 5-Fold `StratifiedKFold(shuffle=True, random_state=42)`.
- **Metrics Evaluated**:
  - `f1_macro`: Unweighted average F1 across all 6 classes.
  - `f1_weighted`: Weighted F1 by class sample frequency.
  - `accuracy`: Global classification accuracy.

---

## 6. Model Comparison Leaderboard (5-Fold Stratified CV)

| Model Architecture | Macro F1 (Unweighted) | Weighted F1 | Accuracy | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest (`class_weight='balanced'`)** | **$\mathbf{0.3093}$** 🏆 | $0.5541$ | $55.85\%$ | Highest Macro F1 (best handling of rare classes 3 & 8). |
| **CatBoost (`auto_class_weights='Balanced'`)** | $0.2973$ | $0.5622$ | $56.88\%$ | Strong balanced multi-class performance. |
| **LightGBM (`class_weight='balanced'`)** | $0.2882$ | $0.5587$ | $57.18\%$ | Fast tree splitting. |
| **Logistic Regression (`class_weight='balanced'`)** | $0.2873$ | $0.4635$ | $42.46\%$ | Linear log-odds boundary. |
| **HistGradientBoosting (`class_weight='balanced'`)** | $0.2864$ | $0.5527$ | $56.44\%$ | Native binned boosting. |
| **XGBoost (`tree_method='hist'`)** | $0.2788$ | **$0.5629$** | **$58.13\%$** 🏆 | **Highest overall global accuracy!** |
| **Decision Tree (`max_depth=8`)** | $0.2592$ | $0.4410$ | $43.05\%$ | Single tree benchmark. |

---

## 7. Strategic Solution: Business Tier Binning (3-Tier Classification)

In real winery operations, wines are categorized into **3 business tiers**:
- **Low / Budget Tier (Ratings 3 & 4)**: Target `0`
- **Standard Table Wine (Ratings 5 & 6)**: Target `1`
- **Premium / Vintage Tier (Ratings 7 & 8)**: Target `2`

### **Empirical Results of 3-Tier Business Reframing:**
- **3-Tier Accuracy**: **$\mathbf{80.50\%}$** (Up from $58.1\%$)!
- **3-Tier Macro F1**: **$\mathbf{0.5544}$** (Up from $0.3093$)!

---

## 8. Confusion Matrix Diagnostic Insights
- **Adjacent Error Clustering**: The 6-class confusion matrix heatmap shows that **$95\%+$ of all misclassifications occur strictly on adjacent classes** ($5 \leftrightarrow 6$ or $6 \leftrightarrow 7$).
- **Zero Catastrophic Failures**: The model never makes severe multi-grade errors (e.g. predicting Class 8 for a Class 3 wine).

---

## 9. Key Lessons Learned
- **Macro F1 vs Accuracy**: Accuracy ($58.1\%$) can mask rare class failure, while Macro F1 ($0.309$) exposes rare class starvation.
- **Adjacent Confusion**: In ordinal rating data (like 1-10 scores), errors are almost exclusively adjacent ($5 \leftrightarrow 6$).
- **Business Reframing**: Collapsing 6 granular subjective ratings into 3 operational business tiers converts a noisy 6-class problem into an **$80.5\%$ accurate decision engine**.
