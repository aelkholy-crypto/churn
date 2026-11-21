import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Customer Churn Analysis", layout="wide")
st.title("Customer Churn Analysis & Prediction")

uploaded_file = st.file_uploader("Upload customer dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(111.csv.csv)
    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    id_cols = [c for c in df.columns if c.lower() in ["id", "customer_id", "customerid"]]
    df_clean = df.drop(columns=id_cols, errors="ignore").copy()

    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")

    if "Churn" in df_clean.columns:
        if df_clean["Churn"].dtype == "object":
            df_clean["Churn"] = df_clean["Churn"].map({"Yes": 1, "No": 0})
        df_clean = df_clean.dropna(subset=["Churn"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", len(df_clean))
        col2.metric("Churn Rate", f"{df_clean['Churn'].mean() * 100:.2f}%")
        col3.metric("Features", df_clean.shape[1] - 1)

        X = df_clean.drop(columns=["Churn"])
        y = df_clean["Churn"].astype(int)

        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]), num_cols),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]), cat_cols),
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = Pipeline([
            ("prep", preprocessor),
            ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42))
        ])
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        st.subheader("Decision Tree Model Performance")
        st.success(f"Test Accuracy: {acc * 100:.2f}%")

        df_clean["Churn_Risk_Score"] = model.predict_proba(X)[:, 1].round(3)
        st.subheader("Top High-Risk Customers")
        st.dataframe(df_clean.sort_values(by="Churn_Risk_Score", ascending=False).head(10))
    else:
        st.warning("Column 'Churn' not found in the uploaded file.")
else:
    st.info("Please upload a CSV file to start.")
