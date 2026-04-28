# Jamboree Education — Graduate Admissions Predictor

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asmi2604/Jamboree_Admission_Analysis/blob/main/Jamboree_Admission_CaseStudy.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-OLS%20Regression-orange.svg)](https://www.statsmodels.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Ridge%20%7C%20Lasso-F7931E.svg)](https://scikit-learn.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13%2B-4C72B0.svg)](https://seaborn.pydata.org/)

End-to-end **graduate admissions probability prediction** pipeline for Jamboree Education.
Transforms raw student academic profiles into a clean, assumption-tested Linear Regression model —
comparing OLS, Ridge, and Lasso regression, validating all five linear regression assumptions,
and surfacing actionable business insights about which factors drive Ivy League admission chances.

---

## Highlights

| Feature | Detail |
| --- | --- |
| **Full EDA pipeline** | Univariate + bivariate analysis with distribution plots, box plots, violin plots, and correlation heatmap |
| **Assumption testing** | All 5 OLS assumptions verified — Multicollinearity (VIF), Mean of Residuals, Linearity, Homoscedasticity, Normality |
| **VIF-based feature reduction** | Iterative drop of high-VIF features (> 5) until multicollinearity is resolved |
| **Three regression models** | OLS (Statsmodels) · Ridge (CV-tuned) · Lasso (CV-tuned, auto feature selection) |
| **Hypothesis testing** | Breusch-Pagan test · Shapiro-Wilk · Kolmogorov-Smirnov · LOWESS residual smoothing |
| **Outlier treatment** | IQR detection across all continuous features + Winsorization (1st–99th percentile) |
| **Model evaluation** | MAE, RMSE, R², Adjusted R² on both train and test sets with overfitting gap check |
| **Business insights** | Feature importance ranking, student segmentation by admission bucket, coaching ROI analysis |

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              Raw Data Layer                 │
                    │   Jamboree_Admission.csv (500 rows × 9 cols)│
                    │   Student academic profiles — Indian        │
                    │   graduate applicants to Ivy League         │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────┐
                    │         Data Cleaning & EDA                 │
                    │  Drop Serial No. · Rename columns           │
                    │  Duplicate check · Missing value check      │
                    │  Research → category dtype                  │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────┐
                    │         Exploratory Data Analysis           │
                    │  Univariate: histograms, KDE, box plots     │
                    │  Bivariate:  scatter + regression lines     │
                    │  Correlation heatmap (Pearson r matrix)     │
                    └──────────┬──────────────────┬──────────────┘
                               │                  │
          ┌────────────────────▼────┐   ┌─────────▼────────────────────┐
          │  Outlier Treatment      │   │   Data Preparation           │
          │  IQR detection          │   │   80/20 train-test split     │
          │  Winsorization          │   │   StandardScaler (Ridge/Lasso)│
          │  6 continuous features  │   │   Unscaled (OLS Statsmodels) │
          └────────────────────┬────┘   └─────────┬────────────────────┘
                               └────────┬─────────┘
                                        │
               ┌────────────────────────┼────────────────────────┐
               │                        │                        │
    ┌──────────▼──────────┐  ┌──────────▼──────────┐  ┌─────────▼──────────────┐
    │  OLS Regression     │  │  Ridge Regression   │  │  Lasso Regression      │
    │  Statsmodels        │  │  CV alpha selection │  │  CV alpha selection    │
    │  Full stats summary │  │  L2 regularisation  │  │  L1 + auto feature     │
    │  p-values · CIs     │  │  Coeff shrinkage    │  │  selection (zeroing)   │
    └──────────┬──────────┘  └──────────┬──────────┘  └─────────┬──────────────┘
               └────────────────────────┼────────────────────────┘
                                        │
                    ┌───────────────────▼─────────────────────────┐
                    │        Assumption Testing (5 checks)        │
                    │  VIF iterative drop · Mean residuals ≈ 0    │
                    │  Residual vs Fitted · Breusch-Pagan         │
                    │  QQ plot · Shapiro-Wilk · K-S test          │
                    └───────────────────┬─────────────────────────┘
                                        │
                    ┌───────────────────▼─────────────────────────┐
                    │     Model Evaluation & Business Insights    │
                    │  MAE · RMSE · R² · Adj R² (Train & Test)    │
                    │  Admission bucket segmentation              │
                    │  Coaching ROI recommendations               │
                    └─────────────────────────────────────────────┘
```

---

## Project Structure

```
jamboree-admission-predictor/
│
├── Jamboree_Admission_CaseStudy.ipynb    ← Main notebook (open in Colab)
├── Jamboree_Admission.csv                ← Dataset (500 students, 8 features)
└── README.md                             ← This file
```

---

### Local

```bash
# 1. Clone the repo
git clone https://github.com/asmi2604/Jamboree_Admission_Analysis.git

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn scipy

# 3. Launch the notebook
jupyter notebook Jamboree_Admission_CaseStudy.ipynb
```

---

## Dataset

| Column | Description | Range |
| --- | --- | --- |
| `Serial No.` | Unique row identifier — dropped before modeling | — |
| `GRE Score` | Graduate Record Examination score | 290–340 |
| `TOEFL Score` | Test of English as a Foreign Language | 92–120 |
| `University Rating` | Prestige of undergraduate institution | 1–5 |
| `SOP` | Statement of Purpose strength | 1–5 |
| `LOR` | Letter of Recommendation strength | 1–5 |
| `CGPA` | Undergraduate GPA | 6.8–9.92 |
| `Research` | Research experience (binary) | 0 / 1 |
| **`Chance of Admit`** | 🎯 **Target variable** — admission probability | 0–1 |

- **500 rows**, 8 features, 1 continuous target
- No missing values, no duplicate rows
- All features numeric — `Research` converted to `category` dtype

---

## Notebook Walkthrough

| # | Section | What It Does |
| --- | --- | --- |
| 1 | Import Libraries | pandas, numpy, seaborn, statsmodels, sklearn, scipy — consistent plot styling |
| 2 | Data Loading & Initial Exploration | Shape, dtypes, missing values, `.describe()` statistical summary |
| 3 | Data Type Conversions | Drop `Serial No.`, rename columns, `Research` → category dtype |
| 4 | Missing Value & Duplicate Treatment | `.duplicated()` check, `.isnull().sum()` verification |
| 5 | Univariate Analysis — Continuous | Histograms + KDE + box plots for GRE, TOEFL, CGPA, SOP, LOR, Admit Chance |
| 6 | Univariate Analysis — Categorical | Annotated bar/count plots for University Rating, SOP, LOR, Research |
| 7 | Bivariate Analysis | Scatter plots + regression lines for each feature vs target; box + violin for Research |
| 8 | Correlation Analysis | Lower-triangle Pearson heatmap — feature-feature and feature-target correlations |
| 9 | Outlier Detection & Treatment | IQR bounds table → Winsorization at 1st–99th percentile |
| 10 | Data Preparation | 80/20 train-test split · `StandardScaler` for Ridge/Lasso · unscaled for OLS |
| 11 | OLS Linear Regression | `statsmodels.OLS` — full summary, coefficient plot with 95% CIs, p-value annotations |
| 12 | **Assumption 1: Multicollinearity** | VIF computed for all features → iterative drop until VIF ≤ 5 → model rebuilt |
| 13 | **Assumption 2: Mean of Residuals** | Direct computation — expected ≈ 0 (OLS mathematical guarantee) |
| 14 | **Assumption 3: Linearity** | Residuals vs Fitted scatter + LOWESS smoothing trend line |
| 15 | **Assumption 4: Homoscedasticity** | Breusch-Pagan test + Scale-Location plot |
| 16 | **Assumption 5: Normality of Residuals** | Histogram + KDE + QQ plot + Shapiro-Wilk + K-S test |
| 17 | Model Evaluation | MAE, RMSE, R², Adj R² on train and test — overfitting gap check |
| 18 | Ridge & Lasso Regression | Cross-validated α · coefficient comparison bar chart · Lasso zeroed features |
| 19 | Model Comparison Dashboard | All three models side-by-side · assumption check summary table |
| 20 | Actionable Insights & Recommendations | Feature importance ranking · student segmentation · coaching strategy · model roadmap |

---

## Assumption Testing Summary

| Assumption | Test Used | Result |
| --- | --- | --- |
| No Multicollinearity | VIF iterative drop (threshold = 5) | ✅ Met — high-VIF features dropped, all remaining VIF ≤ 5 |
| Mean of Residuals ≈ 0 | Direct computation | ✅ Met — value ≈ 0 (OLS mathematical guarantee) |
| Linearity | Residuals vs Fitted + LOWESS | ✅ Met — no systematic curvature in residual plot |
| Homoscedasticity | Breusch-Pagan Test | ✅ Met — p > 0.05, constant variance confirmed |
| Normality of Residuals | QQ Plot + Shapiro-Wilk + K-S | ✅ Approximately met — bell curve + points on QQ diagonal |

---

## Results

Model performance on 500 student records (80 / 20 train-test split):

| Model | Train R² | Test R² | Test MAE | Test RMSE | Overfitting Gap |
| --- | :---: | :---: | :---: | :---: | :---: |
| OLS Linear Regression (VIF-cleaned) | ~0.83 | ~0.82 | ~0.050 | ~0.063 | < 0.02 ✅ |
| Ridge Regression (CV-tuned) | ~0.83 | ~0.82 | ~0.051 | ~0.064 | < 0.02 ✅ |
| Lasso Regression (CV-tuned) | ~0.83 | ~0.82 | ~0.051 | ~0.064 | < 0.02 ✅ |

> Train − Test gap < 0.03 across all models → **No overfitting detected.**
> OLS selected as final model for its full statistical interpretability.

### Feature Importance (by Standardized Coefficient)

| Rank | Feature | Pearson r with Target | Business Interpretation |
| :---: | --- | :---: | --- |
| 1 | **CGPA** | 0.88 | Strongest predictor — 1-point GPA increase yields the largest individual boost to admit chance |
| 2 | **GRE Score** | 0.80 | High ROI for coaching — a 10-point improvement meaningfully raises probability |
| 3 | **TOEFL Score** | 0.79 | Closely tied with GRE; reinforces English proficiency for international programs |
| 4 | **University Rating** | 0.71 | Institutional prestige signals academic environment quality |
| 5 | **LOR Strength** | 0.67 | Strong letters from credible referees outperform a polished SOP |
| 6 | **SOP Strength** | 0.68 | Adds holistic context; less statistically dominant than LOR individually |
| 7 | **Research** | 0.55 | Binary +5–8% boost to admit probability, independent of all other scores |

---

## Key Business Insights

| # | Insight | Recommended Action |
| --- | --- | --- |
| 1 | **CGPA is the #1 predictor** and cannot be retaken after graduation | Target students in Year 1–2 of undergrad with academic coaching — highest ROI intervention |
| 2 | **GRE is fully retakeable** and the 2nd strongest predictor | Position GRE prep as a high-return investment; a 10-point gain = measurable probability lift |
| 3 | **Research adds +5–8%** even after controlling for GRE and CGPA | Launch a research placement service — strong premium upsell with clear data-backed value |
| 4 | **LOR outperforms SOP** in predictive power | Offer LOR coaching workshops — help students select the right referee and brief them effectively |
| 5 | **Student profiles cluster into 4 admission buckets** | Tailor strategy per bucket: < 50% → school selection; 50–70% → test retakes; 70–85% → fine-tuning; > 85% → scholarships |

---

## Tech Stack

| Category | Technology |
| --- | --- |
| Language | Python 3.10 |
| Data manipulation | pandas · NumPy |
| Visualisation | Matplotlib · Seaborn |
| Statistical modelling | Statsmodels (OLS, Breusch-Pagan, VIF) |
| Machine learning | scikit-learn (Ridge, Lasso, StandardScaler, train_test_split, metrics) |
| Statistical testing | SciPy (Shapiro-Wilk, K-S test, probplot) |
| Notebook environment | Google Colab / Jupyter |

---

**Asmita Rajendra**
[LinkedIn](https://www.linkedin.com/in/asmita-r-5b23691a1/)· [GitHub](https://github.com/asmi2604)

*Built as a machine learning case study — Jamboree Education, April 2026*
