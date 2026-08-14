# Day 01 — Decision Log: Ames Housing Price Regression

## Problem Understanding
- **Objective**: Predict residential home sales prices in Ames, Iowa using physical, location, and condition features.
- **Problem Type**: Tabular Regression with mixed feature types (continuous, discrete, ordinal, nominal).
- **Target Variable**: `Sale_Price`
- **Evaluation Metric**: Root Mean Squared Error on log-transformed prices ($\text{RMSE}(\log y, \log \hat{y})$).

## Initial Observations
- 2,930 instances, 81 features.
- Inferred dtypes: 2 float, 33 int, 46 string (object).
- Semantics vs. Storage: Several integer columns are coded nominal classes (`MS_SubClass`) or discrete counts (`Full_Bath`, `Garage_Cars`), while string columns contain structured ordinal quality scales (`Ex`, `Gd`, `TA`, `Fa`, `Po`).
- Identifiers (`PID`, `Order`) carry no generalizable predictive signal and were dropped immediately to prevent memorization/leakage.

## Hypotheses
1. **Target Skewness**: Right-skewed `Sale_Price` (skewness $\approx 1.74$) leads to heteroscedasticity; modeling on $\log(1 + y)$ will normalize residuals and directly align training loss with the evaluation metric.
2. **Outliers**: High-leverage atypical points ($Gr\_Liv\_Area > 4000\text{ sq ft}$ sold $< \$200k$) represent abnormal partial sales that distort linear hyperplanes and must be filtered.
3. **Domain Aggregations**: Real estate prices are driven by composite variables (`Total_SF = Bsmt + 1stFlr + 2ndFlr`, `Total_Bath`, `House_Age = Yr_Sold - Year_Built`) rather than isolated fragmented area measurements.
4. **Regularized Gradient Boosting**: Restricting tree depth (`max_depth=4`) and using conservative learning rates (`0.03`) with subsampling (`0.8`) prevents tree memorization on medium tabular data ($\sim 2.9\text{k}$ rows).

## EDA Findings
- `Sale_Price` right-skewed with long tail up to $\$750,000+$.
- High multicollinearity observed: `Garage_Cars` $\leftrightarrow$ `Garage_Area` ($r=0.89$), `Total_Bsmt_SF` $\leftrightarrow$ `First_Flr_SF` ($r=0.80$), `Gr_Liv_Area` $\leftrightarrow$ `TotRms_AbvGrd` ($r=0.81$).
- Scatter plots revealed 2–3 high-leverage outliers in the bottom right of `Gr_Liv_Area` vs `Sale_Price`.

## Data Quality Issues
- In this OpenML version, structural `NA`s (e.g. `Alley`, `Bsmt_Qual`, `Garage_Type`, `Pool_QC`) were decoded to explicit categories (`'No_Basement'`, `'No_Garage'`, `'No_Alley_Access'`).
- `Misc_Feature` (96.4% missing) and `Mas_Vnr_Type` (60.6% missing) were imputed with explicit category strings (`'Missing'`, `'None'`) rather than naive deletion, retaining signals for luxury amenities.

## Feature Engineering Decisions
- Dropped `PID`, `Order`.
- Filtered `Gr_Liv_Area < 4000`.
- Engineered:
  - `Total_SF = Total_Bsmt_SF + First_Flr_SF + Second_Flr_SF`
  - `Total_Bath = Full_Bath + 0.5 * Half_Bath + Bsmt_Full_Bath + 0.5 * Bsmt_Half_Bath`
  - `House_Age = Yr_Sold - Year_Built`
  - `Remod_Age = Yr_Sold - Year_Remod_Add`

## Preprocessing Decisions
- **Pipeline Architecture**: `ColumnTransformer` with 3 separate branches:
  - **Numeric**: Median imputation + `StandardScaler`.
  - **Ordinal**: Constant imputation + `OrdinalEncoder` for quality/condition attributes.
  - **Nominal**: Constant imputation + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.

## Validation Strategy
- **5-Fold Cross Validation** (`KFold(n_splits=5, shuffle=True, random_state=42)`).
- Leakage prevented by wrapping all preprocessing inside Scikit-learn `Pipeline` objects evaluated within CV folds.

## Baseline & Iteration Progression

| Stage / Iteration | Ridge | Random Forest | HistGradientBoosting | XGBoost |
|---|---|---|---|---|
| **1. Raw Prices ($y$) (Baseline)** | 28,083 | 26,414 | 23,890 | — |
| **2. Log Target $\log(y)$ + Preprocessing** | 0.1339 | 0.1428 | 0.1273 | 0.1360 (default) |
| **3. Log Target + Domain Features + GPU Tuned XGBoost** | — | — | — | **0.1205** |

## Results & Analysis
- **Target Transformation Impact**: Moving to $\log(y)$ transformed multiplicative price multipliers into additive linear relationships, dramatically improving Ridge from being the worst model to beating default Random Forest.
- **Hyperparameter Sensitivity in Boosting**: Default XGBoost (`lr=0.3`, `depth=6`) overfit on small samples ($0.1360$), but controlling tree complexity (`max_depth=4`), lowering step size (`lr=0.03`, `n_estimators=600`), and adding feature/sample subsampling dropped RMSE to **`0.1205`**.

## Final Decision
- Best single model: **Tuned XGBoost Regressor** (`0.1205` Log-RMSE).
- Recommended production solution: Blend of Tuned XGBoost (`70%`) + Ridge (`30%`) to exploit uncorrelated residual errors.

## What Worked
- Log-transformation on skewed target.
- Filtering high-leverage outliers ($Gr\_Liv\_Area > 4000$).
- Domain composite features (`Total_SF`, `Total_Bath`, `House_Age`).
- Leaf/tree regularization + conservative learning rates in gradient boosting.

## What Failed
- Un-regularized default XGBoost with large learning rate ($0.3$).
- Raw dollar regression without variance stabilization.

## Lessons Learned
- Never judge an algorithm family solely by out-of-the-box library default hyperparameters.
- Feature engineering derived from domain knowledge consistently beats blind hyperparameter search.
- Preprocessing pipelines must be scoped strictly to feature semantics (ordinal vs nominal vs numeric).
