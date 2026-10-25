"""Small educational Streamlit interface for the saved heart disease model."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "best_model.joblib"

st.set_page_config(page_title="Heart Disease Risk Classifier", page_icon="♥", layout="centered")
st.title("Heart Disease Prediction Demo")
st.caption("Major project demonstration using the supplied heart dataset")
st.warning(
    "Educational use only. This model is not a medical device and must not be used to diagnose, "
    "screen, or make treatment decisions. It was trained on a small historical dataset."
)

if not MODEL_PATH.exists():
    st.error("The trained model is missing. Run `python train_models.py` from the project folder first.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
numeric_min = bundle["training_data_min"]
numeric_max = bundle["training_data_max"]
category_values = bundle["category_values"]

st.write(
    f"Selected model: **{bundle['model_name']}**. Enter the feature values using the codes "
    "stored in the dataset. Coded categories are shown as numbers because the source file "
    "does not include a complete, verified codebook."
)

with st.form("prediction_form"):
    left, right = st.columns(2)
    values = {}
    with left:
        values["age"] = st.number_input(
            "Age (years)", min_value=float(numeric_min["age"]), max_value=float(numeric_max["age"]),
            value=54.0, step=1.0,
        )
        values["trestbps"] = st.number_input(
            "Resting blood pressure", min_value=float(numeric_min["trestbps"]),
            max_value=float(numeric_max["trestbps"]), value=130.0, step=1.0,
        )
        values["chol"] = st.number_input(
            "Serum cholesterol", min_value=float(numeric_min["chol"]),
            max_value=float(numeric_max["chol"]), value=240.0, step=1.0,
        )
        values["thalach"] = st.number_input(
            "Maximum heart rate achieved", min_value=float(numeric_min["thalach"]),
            max_value=float(numeric_max["thalach"]), value=150.0, step=1.0,
        )
        values["oldpeak"] = st.number_input(
            "Oldpeak", min_value=float(numeric_min["oldpeak"]),
            max_value=float(numeric_max["oldpeak"]), value=1.0, step=0.1,
        )
    with right:
        for feature, label in [
            ("sex", "Sex (dataset code)"),
            ("cp", "Chest pain type (dataset code)"),
            ("fbs", "Fasting blood sugar (dataset code)"),
            ("restecg", "Resting ECG (dataset code)"),
            ("exang", "Exercise-induced angina (dataset code)"),
            ("slope", "ST segment slope (dataset code)"),
            ("ca", "Major vessels (dataset code)"),
            ("thal", "Thal (dataset code)"),
        ]:
            values[feature] = st.selectbox(label, options=category_values[feature], index=0)

    submitted = st.form_submit_button("Run model")

if submitted:
    row = pd.DataFrame([{name: values[name] for name in bundle["features"]}])
    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.5)
    st.subheader("Model output")
    st.metric("Estimated probability for class 1", f"{probability:.1%}")
    if prediction == 1:
        st.info("The model classified this record as class 1 in this dataset.")
    else:
        st.success("The model classified this record as class 0 in this dataset.")
    st.caption("This output reflects dataset labels and a 0.50 threshold, not an individual's clinical risk.")
