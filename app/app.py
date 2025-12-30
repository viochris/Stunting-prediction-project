"""
Module: Streamlit Frontend Application
Author: Silvio Christian, Joe
Description: 
    The user interface layer built with Streamlit. It handles user interactions, 
    input validation, and calls the backend inference engine to display results.
"""

import streamlit as st
import numpy as np
from model import load_model, predict
from preprocess import preprocess_input

# =========================================================
# 1. APP CONFIGURATION & RESOURCE LOADING
# =========================================================
# 'load_model' is called at the top level so resources are cached/loaded 
# when the script runs, ensuring fast performance for the user.
gender_enc, stunting_enc, scaler, best_model = load_model()

# Configuring the browser tab title and layout.
st.set_page_config(page_title="Stunting Prediction", page_icon="🧒", layout="centered")
st.header("🧒 Stunting Prediction App")
st.write("Enter the child's details below to predict the likelihood of stunting.")

# =========================================================
# 2. INPUT FORM CONSTRUCTION
# =========================================================
# Using 'st.form' creates a batching context.
# Without this, the app would re-run the entire script every time the user 
# types a single character in the input fields (inefficient).
with st.form("prediction_form"):
    name = st.text_input("👶 Child's Name", value="Susy")

    # Selectbox restricts input to predefined valid options, preventing user error.
    gender = st.selectbox(
        "🚻 Gender",
        ("Laki-laki", "Perempuan"),
        index=None,
        placeholder="Select gender..."
    )

    # Number inputs with min/max constraints to ensure data validity (Sanity Check).
    age = st.number_input(
        "📅 Age (Months)",
        min_value=0,
        max_value=24, # Constrained to toddlers (0-2 years)
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
        "⚖️ Weight (kg)",
        min_value=0.0,
        value=13.3,
        step=0.1
    )

    # The script waits here until the user clicks 'Predict'.
    submitted = st.form_submit_button("🚀 Predict")

# =========================================================
# 3. EXECUTION LOGIC
# =========================================================
if submitted and gender is not None:
    # 1. Preprocessing: Converting UI inputs -> DataFrame.
    input_value = preprocess_input(gender, age, tinggi_badan, berat_badan)
    
    # 2. Prediction: Running the full ML pipeline.
    result, conf = predict(gender_enc, stunting_enc, scaler, best_model, input_value)

    # 3. Output Display: Showing the result to the user.
    st.markdown("---")
    st.subheader("🔎 Prediction Result")
    st.write(f"**Name:** {name if name else 'N/A'}")

    # Displaying the final classification result clearly with a success banner.
    # Note: 'conf[0]' extracts the number from the array.
    # Note: ':.2%' formats the number into a percentage (e.g., 0.98 becomes 98.00%).
    st.success(f"**Prediction:** {result} with Confidence {conf[0]:.2%}")
else:
    # Error Handling: Prompting the user if the form is submitted incomplete (e.g., missed gender).
    st.warning("⚠️ Oops! Looks like some fields are missing. Please fill in all inputs first.")
