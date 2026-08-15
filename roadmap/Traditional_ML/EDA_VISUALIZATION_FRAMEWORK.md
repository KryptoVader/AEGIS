# AEGIS — World-Class EDA Visualization Framework

The goal of this framework is to establish and enforce **professional, reasoning-driven EDA visualization** across all AEGIS projects.

---

## Core Philosophy

Treat visualization as an evolution of questions.

Never start with:
> *"Which Seaborn plot should I use?"*

Start with:
> *"What question am I trying to answer about the data?"*

Then determine:
1. **What question am I asking?**
2. **What type of variables are involved?** (Numerical, Categorical, Ordinal, Datetime, Target)
3. **What visual representation naturally answers that question?**
4. **What does that visualization reveal?**
5. **What limitations does it have?**
6. **What question does the next visualization help answer?**
7. **What EDA or ML decision can be made from the observation?**

The visualization workflow must feel like an **investigation** rather than a checklist.

---

## 1. Visualization Evolution

Visualization follows this logical progression:

$$\text{RAW DATA} \longrightarrow \text{Variable Types} \longrightarrow \text{Univariate} \longrightarrow \text{Bivariate} \longrightarrow \text{Multivariate} \longrightarrow \text{Target-Centric} \longrightarrow \text{Scale Limits} \longrightarrow \text{ML Decisions}$$

Each stage naturally emerges from the limitations of the previous stage:
- **Numerical variable**:
  $$\text{Histogram (binned)} \longrightarrow \text{KDE (smooth estimate)} \longrightarrow \text{Box Plot (quartile/tail summary)} \longrightarrow \text{ECDF (bin-free cumulative distribution)}$$

---

## 2. Univariate Numerical Visualization Progression

### Histogram
- **Question**: *What is the frequency distribution across numeric ranges?*
- **Inspect**: Bins, bin width, center, spread, skewness, multimodality, tails.
- **Limitation**: Different bin widths can alter visual interpretation.

### KDE (Kernel Density Estimate)
- **Question**: *Can I get a smooth continuous estimate of the underlying probability density?*
- **Inspect**: Density shape, modes, bandwidth smoothing.
- **Limitation**: Smoothing can create artifacts (e.g. density below zero for strictly positive variables) and hide true spikes.

### Box Plot
- **Question**: *Where are the median, IQR, quartiles, and extreme observations?*
- **Inspect**: Median ($Q2$), IQR ($Q3 - Q1$), whiskers ($1.5 \times \text{IQR}$).
- **Principle**: Points beyond whiskers are observations flagged by a statistical convention, not automatically errors.

### ECDF (Empirical Cumulative Distribution Function)
- **Question**: *What proportion of observations fall at or below any threshold $x$?*
- **Core Formula**: $\text{ECDF}(x) = \frac{\# \{X_i \le x\}}{N} = P(X \le x)$.
- **Strengths**: Bin-free, preserves 100% of data points, reveals percentiles, concentration, and enables precise distribution comparisons without binning artifacts.

---

## 3. Univariate Categorical Visualization

- **Question**: *What discrete categories exist and how frequently do they occur?*
- **Tools**: Count plots, frequency tables, normalized proportions.
- **Inspect**: Cardinality, imbalance, rare categories, long-tail distributions.
- **ML Impact**: Informs one-hot vs target vs frequency encoding, risk of unseen categories in test folds.

---

## 4. Bivariate Visualization Progression

### Numerical $\times$ Numerical
- **Question**: *How does feature $X$ relate to feature $Y$?*
- **Tools**: Scatter plots, regression trendlines, log-transforms, hexbin/2D density for large datasets.
- **Inspect**: Linearity, non-linearity, heteroscedasticity, clusters, multicollinearity ($r \approx 1$).
- **Rule**: Correlation captures linear association only; zero linear correlation does not imply independence.

### Numerical $\times$ Categorical
- **Question**: *How does the numerical distribution vary across categories/groups?*
- **Tools**: Box plots $\longrightarrow$ Violin plots $\longrightarrow$ Strip/Swarm plots (for small $N$).
- **Trade-off**: Box plot gives robust quartiles; Violin adds density contours; Strip plots show actual sample points.

### Categorical $\times$ Categorical
- **Question**: *Are the categorical variables associated or independent?*
- **Tools**: Contingency tables (cross-tabs), normalized grouped/stacked bar charts, heatmaps.
- **Principle**: Always check normalized proportions alongside raw counts to avoid sample size bias.

---

## 5. Multivariate Visualization

- **Question**: *How does the relationship between $X$ and $Y$ change when a third variable $Z$ is introduced?*
- **Tools**: Hue/style grouping, faceted subplots (`FacetGrid`), correlation matrices, interaction plots.
- **Inspect**: Confounders, interaction effects, Simpson's Paradox, subgroup shifts.

---

## 6. Target-Centric Visualization

- **Question**: *Does feature $X$ separate or shift across target classes / values?*
- **Classification Target**:
  - *Numerical feature*: Compare distributions per class via split KDEs, comparative ECDFs, boxplots by class.
  - *Categorical feature*: Churn/positive rate per category, cross-tabulated proportions.
- **ML Impact**: Identifies candidate predictive signals, non-linear boundaries, and feature interactions.

---

## 7. Large Dataset Visualization

- **Rule**: For large sample sizes ($N > 50,000$), avoid naive scatter plots (overplotting hides density).
- **Evolution**:
  $$\text{Individual Points (Overplotting)} \longrightarrow \text{Subsampling / Alpha Blending} \longrightarrow \text{Hexbin / 2D Density / Contours}$$

---

## 8. The 7-Step Visualization Standard

Every visualization generated must answer and document:

1. **Question**: What specific question are we asking?
2. **Why this visual**: Why is this representation appropriate over alternatives?
3. **What to look for**: Which patterns, tails, or shifts matter?
4. **Interpretation**: What does the observed pattern mean?
5. **Limitations**: What does this plot hide or distort?
6. **Next Question**: What question does this naturally lead to?
7. **ML / EDA Implication**: What concrete cleaning, engineering, or modeling decision follows?

---

## 9. Decision Pipeline

$$\text{Data} \longrightarrow \text{Question} \longrightarrow \text{Plot} \longrightarrow \text{Observation} \longrightarrow \text{Interpretation} \longrightarrow \text{Hypothesis} \longrightarrow \text{EDA Decision} \longrightarrow \text{ML Decision}$$
