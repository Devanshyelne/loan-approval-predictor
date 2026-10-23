# Loan Approval Predictor

A Python mini-project that predicts loan approval from applicant data, with an interactive Streamlit dashboard for exploration, evaluation and prediction.

## Project structure

```
loan-approval-predictor/
├── app.py                          # Streamlit app (training pipeline + UI)
├── loan-approval-predictor.ipynb   # Original exploratory notebook (unchanged)
├── loan_approval_data.csv          # Dataset (place it here)
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

```bash
python -m venv .venv
```

Activate the environment:

- macOS / Linux: `source .venv/bin/activate`
- Windows PowerShell: `.venv\Scripts\Activate.ps1`

Install dependencies and launch:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Using your data

The app reads `loan_approval_data.csv` from the project folder, or a CSV you upload in the sidebar. Pick the approval column in the sidebar (it is pre-selected if named `Loan_Approved`, `Loan_Status`, `Approved` or `target`). The target must have exactly two classes, such as Yes/No. Columns whose names end in `ID` are excluded from training.

## Sections

- **Overview:** dataset size, approval rate, held-out accuracy, class distribution and top model drivers.
- **Predict:** applicant form built from the dataset's own columns; shows the predicted class and its probability.
- **Data Explorer:** data preview, missing values, distributions by outcome and numeric correlations.
- **Model Performance:** held-out test metrics, confusion matrix, classification report, cross-validation summary and feature importance.
- **About:** methodology, limitations and run instructions.

## Methodology

1. Load the CSV and encode the target as 0/1.
2. Split 80/20 with stratification and a fixed seed (`random_state=42`).
3. Build one scikit-learn `Pipeline`: median imputation and scaling for numeric columns; most-frequent imputation and one-hot encoding (with unknown-category handling) for categorical columns; then `GradientBoostingClassifier`.
4. Run 5-fold stratified cross-validation on the training split only, then fit on the full training split.
5. Evaluate on the held-out test set. Cross-validation and test scores are reported separately.

Training is cached with `st.cache_resource`, so the model retrains only when the data or target column changes. No serialized model files are required.

## Limitations

- This is a learning project, not a regulated credit-scoring system.
- Scores depend on the dataset. Small or synthetic data can produce misleading results.
- Feature importance does not establish causation or fairness.
- Do not enter real personal or financial information, and do not use predictions for real lending decisions.
