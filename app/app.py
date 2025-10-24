import streamlit as st
import numpy as np
from model import load_model, predict
from preprocess import preprocess_input

# ==============================
# Load Model & Encoders
# ==============================
gender_enc, stunting_enc, scaler, best_model = load_model()

# ==============================
# App Header
# ==============================
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")
st.header("🩺 Diabetes Prediction App")
st.write("Enter the patient's details below to predict the likelihood of diabetes.")

# ==============================
# Input Form
# ==============================
with st.form("prediction_form"):
    name = st.text_input("👤 Patient Name", value="Susy")

    gender = st.selectbox(
        "🤰 Gender",
        ("Laki-laki", "Perempuan"),
        index=None,
        placeholder="Select gender..."
    )

    age = st.number_input(
        "🍬 Age (Months)",
        min_value=0,
        max_value=24,
        value=19,
        step=1
    )

    tinggi_badan = st.number_input(
        "📏 Height (cm)",
        min_value=0.0,
        value=91.6,
        step=0.1
    )

    berat_badan = st.number_input(
        "🩸 Weight (kg)",
        min_value=0.0,
        value=13.3,
        step=0.1
    )

    submitted = st.form_submit_button("🚀 Predict")


# ==============================
# Prediction Result
# ==============================
if submitted and gender is not None:
    input_value = preprocess_input(gender, age, tinggi_badan, berat_badan)
    result = predict(gender_enc, stunting_enc, scaler, best_model, input_value)

    st.markdown("---")
    st.subheader("🔎 Prediction Result")
    st.write(f"**Name:** {name if name else 'N/A'}")

    # display prediction result
    st.success(f"**Prediction:** {result}")
else:
    st.warning("🩺 Oops! Looks like some fields are missing. Please fill in all inputs first.")
