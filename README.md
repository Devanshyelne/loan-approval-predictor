# Loan Approval Predictor

My first end-to-end ML project — a binary classifier that predicts whether a loan application will be approved, built from raw, messy data through to a properly validated model.

Built the full pipeline from scratch and then iterated on it independently: fixing data leakage, adding stratified splits, tuning models to reduce overfitting, and validating everything with cross-validation.

## Problem Statement

Given an applicant's financial and demographic details (income, credit score, DTI ratio, employment status, etc.), predict whether their loan will be approved (`Yes` / `No`).

## Dataset

- 1000 samples, 20 raw features (income, credit score, DTI ratio, collateral value, employment status, marital status, etc.)
- ~5% missing values across all columns (handled via imputation)
- Mild class imbalance, handled with stratified train/test split

## Approach

1. **EDA** — class balance, distribution plots, boxplots for outliers, correlation heatmap to identify predictive features (Credit Score and DTI Ratio turned out to be the strongest predictors)
2. **Preprocessing** — missing value imputation (mean/mode), label encoding for ordinal features, one-hot encoding for nominal features, feature scaling (StandardScaler)
3. **Feature engineering** — added squared transforms of Credit Score and DTI Ratio to capture non-linear effects
4. **Modeling** — trained and compared 8 classifiers: Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM, Random Forest, Gradient Boosting, XGBoost
5. **Validation** — 5-fold stratified cross-validation on the training set to get a reliable performance estimate, *before* touching the test set
6. **Final evaluation** — best model evaluated once on the held-out test set

## Key Fixes & Learnings

This project went through a real debugging pass, not just a first-draft run:

- **Data leakage fixed** — imputation and encoding were originally fit on the full dataset before splitting; moved to fit-on-train-only
- **Stratified splitting** added to preserve class balance across train/test
- **Reproducibility** — `random_state` set consistently across all models
- **Overfitting addressed** — tree-based models (Decision Tree, Random Forest, Gradient Boosting, XGBoost) were initially hitting 100% train accuracy. Fixed by reducing `max_depth`, increasing `min_samples_leaf`, adding `subsample`/regularization
- **Cross-validation added** to confirm the overfitting fix actually worked, and to compare models fairly instead of relying on a single lucky train/test split

## Results (5-fold Cross-Validated, on training data)

| Model | CV Accuracy | CV F1 Score |
|---|---|---|
| **Gradient Boosting** | **94.6%** | **91.4%** ✅ best |
| XGBoost | 94.5% | 91.1% |
| Decision Tree | 94.3% | 90.9% |
| Random Forest | 91.9% | 85.6% |
| Logistic Regression | 87.3% | 77.7% |
| SVM | 87.1% | 77.0% |
| Naive Bayes | 86.3% | 75.0% |
| KNN | 76.5% | 56.4% |

**Best model: Gradient Boosting Classifier** — highest and most stable cross-validated F1 score, confirmed on the held-out test set afterward.

## Tech Stack

Python, scikit-learn, XGBoost, pandas, numpy, matplotlib, seaborn

## Project Structure

```
loan-approval-predictor.ipynb   # full pipeline: EDA → preprocessing → modeling → CV → evaluation
loan_approval_data.csv          # dataset
README.md
```

## How to Run

```bash
git clone https://github.com/Devanshyelne/loan-approval-predictor.git
cd loan-approval-predictor
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
jupyter notebook loan-approval-predictor.ipynb
```

## Future Work

- Deploy as an interactive web app (Streamlit) for live predictions
- Hyperparameter tuning via GridSearchCV/Optuna instead of manual tuning
- SHAP values for model interpretability

---

Built by [Devansh Yelne](https://github.com/Devanshyelne) — 2nd year AI & ML student, learning ML by building and breaking things properly.
