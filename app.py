"""Loan Approval Predictor - Streamlit mini-project.

Trains a scikit-learn pipeline on the project's CSV (no saved model files needed),
caches it, and offers data exploration, model evaluation and applicant prediction.
"""
from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "loan_approval_data.csv"
RANDOM_STATE = 42
TARGET_CANDIDATES = ["Loan_Approved", "Loan_Status", "Approved", "target"]
POSITIVE_LABELS = {"yes", "y", "approved", "approve", "accepted", "1", "true"}
ID_PATTERN = re.compile(r"(^|_)id$", re.IGNORECASE)

st.set_page_config(page_title="Loan Approval Predictor", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1280px; }
    .kpi { border: 1px solid #2b3a55; border-radius: 12px; padding: 1rem 1.1rem;
           background: rgba(30, 41, 66, 0.35); }
    .kpi-label { font-size: .78rem; letter-spacing: .06em; text-transform: uppercase; opacity: .7; }
    .kpi-value { font-size: 1.7rem; font-weight: 700; margin-top: .2rem; }
    .kpi-note { font-size: .78rem; opacity: .6; margin-top: .2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def kpi(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def show_fig(fig: plt.Figure) -> None:
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def encode_target(labels: pd.Series) -> tuple[pd.Series, dict[int, str]]:
    """Map a two-class label column to 0/1. The positive class is the 'approved' label."""
    cleaned = labels.astype(str).str.strip().str.lower()
    classes = sorted(cleaned.unique())
    if len(classes) != 2:
        raise ValueError(f"Target must have exactly 2 classes, found {len(classes)}: {classes}")
    positive = next((c for c in classes if c in POSITIVE_LABELS), classes[1])
    negative = classes[0] if positive == classes[1] else classes[1]
    mapping = {negative: 0, positive: 1}
    return cleaned.map(mapping).astype(int), {0: negative.title(), 1: positive.title()}


def prepare(df: pd.DataFrame, target: str):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(subset=[target]).reset_index(drop=True)
    y, labels = encode_target(df[target])
    dropped = [c for c in df.columns if c != target and ID_PATTERN.search(c)]
    X = df.drop(columns=[target] + dropped)
    X = X.loc[:, X.notna().any()]  # drop completely empty columns
    return X, y, labels, dropped


@st.cache_data(show_spinner=False)
def load_csv(csv_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(csv_bytes))


@st.cache_resource(show_spinner="Training model on your dataset...")
def train_model(csv_bytes: bytes, target: str) -> dict:
    df = load_csv(csv_bytes)
    X, y, labels, dropped = prepare(df, target)
    if len(y) < 20 or y.value_counts().min() < 5:
        raise ValueError("Need at least 20 rows and 5 rows per class to train and evaluate.")

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), num_cols),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
        ],
        remainder="drop",
    )
    model = Pipeline([
        ("pre", preprocessor),
        ("clf", GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3,
            subsample=0.8, min_samples_leaf=5, random_state=RANDOM_STATE)),
    ])

    # Held-out split. Preprocessing is fitted inside the pipeline, so only training rows are used.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # Cross-validation on the training split only (honest estimate, separate from the test set).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    names = model.named_steps["pre"].get_feature_names_out()
    importances = pd.Series(model.named_steps["clf"].feature_importances_, index=names)

    return {
        "model": model,
        "X": X, "y": y, "labels": labels,
        "num_cols": num_cols, "cat_cols": cat_cols, "dropped": dropped,
        "train_size": len(y_train), "test_size": len(y_test),
        "cv_scores": cv_scores,
        "test": {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "confusion": confusion_matrix(y_test, y_pred, labels=[0, 1]),
            "report": classification_report(
                y_test, y_pred, labels=[0, 1],
                target_names=[labels[0], labels[1]], output_dict=True, zero_division=0),
        },
        "importances": importances.sort_values(ascending=False),
    }


# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
st.sidebar.title("Loan Approval Predictor")
uploaded = st.sidebar.file_uploader("Upload a CSV (optional)", type=["csv"])

if uploaded is not None:
    csv_bytes = uploaded.getvalue()
    source = f"Uploaded file: {uploaded.name}"
elif DATA_PATH.exists():
    csv_bytes = DATA_PATH.read_bytes()
    source = f"Project dataset: {DATA_PATH.name}"
else:
    st.error(
        f"Dataset not found at `{DATA_PATH.name}`. Place it in the project folder "
        "next to app.py, or upload a CSV from the sidebar. The file needs a binary "
        "approval column (for example `Loan_Approved` with Yes/No) and feature columns."
    )
    st.stop()

raw = load_csv(csv_bytes)
target_guess = next((c for c in raw.columns if str(c).strip() in TARGET_CANDIDATES), raw.columns[-1])
target = st.sidebar.selectbox("Target column", raw.columns.tolist(),
                              index=list(raw.columns).index(target_guess))

try:
    bundle = train_model(csv_bytes, target)
except Exception as exc:  # show a readable error instead of a traceback
    st.error(f"Could not train on this dataset: {exc}")
    st.stop()

page = st.sidebar.radio("Section", ["Overview", "Predict", "Data Explorer", "Model Performance", "About"])
st.sidebar.caption(source)
st.sidebar.caption("Educational estimates only. Not for real lending decisions.")

X, y, labels = bundle["X"], bundle["y"], bundle["labels"]
test, cv_scores = bundle["test"], bundle["cv_scores"]

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Loan Approval Predictor")
    st.write(
        "This project predicts whether a loan application is likely to be approved, based on "
        "applicant and loan attributes. It trains a Gradient Boosting classifier on historical "
        "records and reports how well it generalises to unseen data."
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Applications", f"{len(y):,}", "Rows used")
    with c2: kpi("Input features", f"{X.shape[1]}", f"{len(bundle['num_cols'])} numeric · {len(bundle['cat_cols'])} categorical")
    with c3: kpi("Approval rate", f"{y.mean():.1%}", f"Class '{labels[1]}'")
    with c4: kpi("Held-out accuracy", f"{test['accuracy']:.1%}", f"{bundle['test_size']} unseen rows")

    left, right = st.columns(2)
    with left:
        st.subheader("Class distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        counts = y.map(labels).value_counts()
        sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index, legend=False, palette="Blues_d")
        ax.set_ylabel("Applications"); ax.set_xlabel("")
        show_fig(fig)
    with right:
        st.subheader("Top model drivers")
        top = bundle["importances"].head(10).sort_values()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(top.index.str.replace(r"^(num|cat)__", "", regex=True), top.values, color="#4c78a8")
        ax.set_xlabel("Relative importance")
        show_fig(fig)
    st.caption("Importance shows what the fitted model relies on. It does not prove causation or fairness.")

elif page == "Predict":
    st.title("Applicant prediction")
    st.warning("Academic demonstration only. Do not enter real personal or financial data.")
    with st.form("applicant"):
        values = {}
        cols = st.columns(3)
        for i, col in enumerate(X.columns):
            container = cols[i % 3]
            s = X[col]
            label = col.replace("_", " ")
            if col in bundle["num_cols"]:
                lo, hi, med = float(s.min()), float(s.max()), float(s.median())
                if pd.api.types.is_integer_dtype(s):
                    values[col] = container.number_input(label, min_value=int(lo), max_value=int(max(hi, lo + 1)), value=int(med), step=1)
                else:
                    values[col] = container.number_input(label, min_value=lo, max_value=max(hi, lo + 1e-6), value=med)
            else:
                options = sorted(s.dropna().astype(str).unique().tolist()) or ["Unknown"]
                values[col] = container.selectbox(label, options)
        submitted = st.form_submit_button("Predict approval", use_container_width=True)

    if submitted:
        # Same column order as training, so the pipeline preprocesses identically.
        row = pd.DataFrame([values], columns=X.columns)
        proba = bundle["model"].predict_proba(row)[0]
        p_pos = float(proba[list(bundle["model"].classes_).index(1)])
        pred = int(p_pos >= 0.5)
        a, b = st.columns([1.4, 1])
        with a:
            if pred == 1:
                st.success(f"Model estimate: **{labels[1]}**")
            else:
                st.error(f"Model estimate: **{labels[0]}**")
            st.progress(p_pos, text=f"Probability of '{labels[1]}': {p_pos:.1%}")
            st.caption("This is a model estimate from historical patterns, not a guaranteed decision.")
        with b:
            kpi("Probability", f"{p_pos:.1%}", f"Threshold 50%")

elif page == "Data Explorer":
    st.title("Data explorer")
    st.dataframe(raw.head(100), hide_index=True)
    m1, m2, m3 = st.columns(3)
    with m1: kpi("Rows", f"{len(raw):,}")
    with m2: kpi("Missing cells", f"{int(raw.isna().sum().sum()):,}")
    with m3: kpi("Duplicate rows", f"{int(raw.duplicated().sum()):,}")

    st.subheader("Missing values")
    missing = raw.isna().sum()
    missing = missing[missing > 0].sort_values()
    if missing.empty:
        st.success("No missing values.")
    else:
        fig, ax = plt.subplots(figsize=(7, max(2.5, 0.35 * len(missing))))
        ax.barh(missing.index, missing.values, color="#e0ac71")
        ax.set_xlabel("Missing cells")
        show_fig(fig)

    st.subheader("Distributions by outcome")
    feature = st.selectbox("Feature", X.columns.tolist())
    df_plot = X.copy()
    df_plot["Outcome"] = y.map(labels)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    if feature in bundle["num_cols"]:
        sns.histplot(data=df_plot, x=feature, hue="Outcome", bins=30, kde=True, element="step", ax=ax)
    else:
        sns.countplot(data=df_plot, x=feature, hue="Outcome", ax=ax)
        ax.tick_params(axis="x", rotation=30)
    show_fig(fig)

    if len(bundle["num_cols"]) >= 2:
        st.subheader("Correlation between numeric features")
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(X[bundle["num_cols"]].corr(), cmap="vlag", center=0, annot=len(bundle["num_cols"]) <= 12, fmt=".2f", ax=ax)
        show_fig(fig)

elif page == "Model Performance":
    st.title("Model performance")
    st.write(
        f"Trained on **{bundle['train_size']:,}** rows. Scores below come from the **held-out test set "
        f"of {bundle['test_size']:,} rows**, which the model never saw during fitting."
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Test accuracy", f"{test['accuracy']:.1%}")
    with c2: kpi("Test precision", f"{test['precision']:.1%}")
    with c3: kpi("Test recall", f"{test['recall']:.1%}")
    with c4: kpi("Test F1", f"{test['f1']:.1%}")
    st.caption(
        f"5-fold cross-validation on the training split (accuracy): "
        f"{cv_scores.mean():.1%} ± {cv_scores.std():.1%}. Reported separately from the test set."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix (test set)")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(test["confusion"], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=[labels[0], labels[1]], yticklabels=[labels[0], labels[1]], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        show_fig(fig)
    with right:
        st.subheader("Classification report")
        st.dataframe(pd.DataFrame(test["report"]).T.round(3))

    st.subheader("Feature importance")
    top_n = st.slider("Features to show", 5, min(25, len(bundle["importances"])), 10)
    top = bundle["importances"].head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * top_n)))
    ax.barh(top.index.str.replace(r"^(num|cat)__", "", regex=True), top.values, color="#4c78a8")
    ax.set_xlabel("Importance")
    show_fig(fig)

elif page == "About":
    st.title("About this project")
    st.markdown(
        """
**Problem.** Predict whether a loan application is approved, to explore how applicant attributes relate to decisions.

**Data.** The CSV in the project folder (or the file you upload). The target column is chosen in the sidebar. Identifier columns (for example `Applicant_ID`) are excluded from training.

**Preprocessing.** Numeric features: median imputation and standard scaling. Categorical features: most-frequent imputation and one-hot encoding. All steps live inside one scikit-learn `Pipeline`, so training and prediction use the same logic.

**Method.** Stratified 80/20 train-test split with a fixed random seed. Cross-validation runs on the training split only. Preprocessing is fitted within each fold, which prevents data leakage.

**Model.** Gradient Boosting Classifier.

**Limitations.** Results depend on the dataset and may reflect bias in historical decisions. Small or synthetic data can give misleading scores. This tool must not be used for real credit decisions.
        """
    )
    st.subheader("Run locally")
    st.code("python -m venv .venv\n.venv\\Scripts\\activate   # Windows\nsource .venv/bin/activate  # macOS/Linux\n"
            "pip install -r requirements.txt\npython -m streamlit run app.py", language="bash")
