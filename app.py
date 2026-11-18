import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

st.title("💳 Credit Card Fraud Detection")
st.write("Upload a transaction dataset, explore it, and train a fraud-detection model.")

uploaded_file = st.sidebar.file_uploader("Upload credit-card CSV file", type=["csv"])
default_path = "creditcard_1_1.csv"

@st.cache_data
def load_data(file, path):
    if file is not None:
        return pd.read_csv(file)
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

df = load_data(uploaded_file, default_path)

if df is None:
    st.info(
        f"Upload your CSV file using the sidebar. Alternatively, place "
        f"`{default_path}` in the same folder as `app.py`."
    )
    st.stop()

if "Class" not in df.columns:
    st.error("The uploaded CSV must contain a target column named 'Class' (0 = non-fraud, 1 = fraud).")
    st.stop()

st.subheader("1. Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Total transactions", f"{len(df):,}")
c2.metric("Number of features", len(df.columns) - 1)
c3.metric("Missing values", int(df.isna().sum().sum()))

with st.expander("Preview dataset"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Dataset information"):
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data type": df.dtypes.astype(str).values,
        "Missing values": df.isna().sum().values,
        "Unique values": df.nunique().values
    })
    st.dataframe(info_df, use_container_width=True)
    st.dataframe(df.describe().T, use_container_width=True)

st.subheader("2. Exploratory data analysis")
class_counts = df["Class"].value_counts().reindex([0, 1], fill_value=0)
class_plot = pd.DataFrame({
    "Transaction type": ["Non-Fraud", "Fraud"],
    "Count": [int(class_counts.loc[0]), int(class_counts.loc[1])]
})
col1, col2 = st.columns(2)
with col1:
    fig = px.bar(class_plot, x="Transaction type", y="Count",
                 title="Class distribution", color="Transaction type")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    if "Amount" in df.columns:
        fig = px.histogram(df, x="Amount", color="Class", nbins=40,
                           title="Transaction amount distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Amount column found; skipping amount chart.")

numeric_df = df.select_dtypes(include=np.number)
if numeric_df.shape[1] > 1:
    with st.expander("Show correlation heatmap"):
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=False, aspect="auto",
                        color_continuous_scale="RdBu_r",
                        title="Feature correlation matrix")
        st.plotly_chart(fig, use_container_width=True)

st.subheader("3. Train machine learning model")
st.caption("Rows with missing values are removed. SMOTE is applied only to training folds/data, never to the test set.")

data = df.dropna().copy()
if data.empty:
    st.error("No rows remain after removing missing values.")
    st.stop()

try:
    X = data.drop(columns=["Class"])
    y = data["Class"].astype(int)
    X = X.select_dtypes(include=np.number)
    if X.shape[1] == 0:
        st.error("No numeric feature columns were found.")
        st.stop()
    if y.nunique() != 2 or not set(y.unique()).issubset({0, 1}):
        st.error("The Class column must contain both classes encoded as 0 and 1.")
        st.stop()
    if y.value_counts().min() < 2:
        st.error("At least two examples of each class are required for a stratified train/test split.")
        st.stop()
except (ValueError, TypeError) as exc:
    st.error(f"Could not prepare the data: {exc}")
    st.stop()

model_choice = st.sidebar.selectbox("Choose model", ["Logistic Regression", "XGBoost"])
test_size = st.sidebar.slider("Test set size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
run_cv = st.sidebar.checkbox("Run 5-fold cross-validation", value=False)

if st.button("Train model", type="primary"):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # SMOTE needs at least two minority-class examples in each training fold.
        minority_count = int(y_train.value_counts().min())
        if minority_count < 2:
            st.error("Not enough minority-class training examples for SMOTE.")
            st.stop()
        k_neighbors = min(5, minority_count - 1)
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)

        if model_choice == "Logistic Regression":
            estimator = ImbPipeline(steps=[
                ("scaler", StandardScaler()),
                ("smote", smote),
                ("model", LogisticRegression(max_iter=2000, random_state=42))
            ])
        else:
            estimator = ImbPipeline(steps=[
                ("smote", smote),
                ("model", XGBClassifier(
                    n_estimators=250,
                    learning_rate=0.1,
                    max_depth=4,
                    min_child_weight=3,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    gamma=1,
                    reg_alpha=2,
                    reg_lambda=2,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1
                ))
            ])

        with st.spinner("Training model..."):
            estimator.fit(X_train, y_train)
            predictions = estimator.predict(X_test)

        st.success("Model training completed.")
        st.subheader("4. Model evaluation")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy_score(y_test, predictions):.4f}")
        m2.metric("Precision", f"{precision_score(y_test, predictions, zero_division=0):.4f}")
        m3.metric("Recall", f"{recall_score(y_test, predictions, zero_division=0):.4f}")
        m4.metric("F1-score", f"{f1_score(y_test, predictions, zero_division=0):.4f}")

        col1, col2 = st.columns(2)
        with col1:
            cm = confusion_matrix(y_test, predictions, labels=[0, 1])
            fig = px.imshow(
                cm, text_auto=True, aspect="auto",
                x=["Predicted Non-Fraud", "Predicted Fraud"],
                y=["Actual Non-Fraud", "Actual Fraud"],
                color_continuous_scale="Blues",
                title="Confusion matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            report = classification_report(
                y_test, predictions, labels=[0, 1],
                target_names=["Non-Fraud", "Fraud"],
                output_dict=True, zero_division=0
            )
            st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

        if run_cv:
            st.write("5-fold cross-validation (SMOTE is fitted separately inside each training fold):")
            min_class_count = int(y.value_counts().min())
            if min_class_count < 5:
                st.warning("Not enough examples in the smallest class for 5-fold stratified cross-validation.")
            else:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                with st.spinner("Running cross-validation..."):
                    scores = cross_val_score(estimator, X, y, cv=cv, scoring="recall", n_jobs=1)
                st.write(f"Mean cross-validation recall: **{scores.mean():.4f}**")
                st.write("Fold recall scores:", [round(float(s), 4) for s in scores])

        st.caption("For fraud detection, pay particular attention to fraud-class recall and precision, not accuracy alone.")
    except Exception as exc:
        st.error(f"Model training failed: {exc}")
        st.exception(exc)

st.sidebar.markdown("---")
st.sidebar.caption("Expected target column: Class (0 = non-fraud, 1 = fraud)")
