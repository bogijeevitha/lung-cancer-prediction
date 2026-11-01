import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Lung Cancer Prediction", page_icon="🫁")

st.title("🫁 Lung Cancer Prediction")
st.subheader("Using Machine Learning")

st.warning("Educational project only. Not intended for medical diagnosis.")

@st.cache_resource
def train_model():
    df = pd.read_csv("lung_cancer_examples.csv")

    df.columns = df.columns.str.strip().str.lower()

    target = "result"
    if target not in df.columns:
        raise ValueError("The dataset must contain a 'result' column.")

    df = df.drop(columns=["name", "surname"], errors="ignore")
    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    transformers = []

    if numeric:
        transformers.append(
            ("num", SimpleImputer(strategy="median"), numeric)
        )

    if categorical:
        transformers.append(
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical)
        )

    preprocess = ColumnTransformer(transformers)

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=100, random_state=42
        ))
    ])

    model.fit(X, y)
    return model, X, numeric, categorical

try:
    model, X, numeric, categorical = train_model()

    st.write("Enter the details below:")

    values = {}

    for col in numeric:
        values[col] = st.number_input(
            col.replace("_", " ").title(),
            value=float(X[col].median())
        )

    for col in categorical:
        choices = X[col].dropna().unique().tolist()
        values[col] = st.selectbox(
            col.replace("_", " ").title(), choices
        )

    if st.button("Predict"):
        input_data = pd.DataFrame([values])
        prediction = model.predict(input_data)[0]

        st.success(f"Model prediction: {prediction}")

except Exception as e:
    st.error(f"Error: {e}")
