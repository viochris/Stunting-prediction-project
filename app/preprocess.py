import pandas as pd

def preprocess_input(gender, age, berat_badan, tinggi_badan):
    """
    Transforms raw user inputs into a structured DataFrame compatible with the model.
    """
    # ---------------------------------------------------------
    # DATA STRUCTURE ALIGNMENT
    # ---------------------------------------------------------
    # Machine Learning models are sensitive to input structure. 
    # We must recreate the exact same schema (column names & order) used during training.
    # Any mismatch in column names here will raise a "Feature Mismatch Error" in sklearn.
    
    df = pd.DataFrame({
        "Jenis Kelamin": [gender],      # Raw string input (e.g., "Laki-laki")
        "Umur (bulan)": [age],          # Numerical input
        "Tinggi Badan (cm)": [tinggi_badan], 
        "Berat Badan (kg)": [berat_badan]
    })
    
    # Returns a single-row DataFrame ready for the Encoding & Scaling pipeline.
    return df
