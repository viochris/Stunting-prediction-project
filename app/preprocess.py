import pandas as pd

def preprocess_input(gender, age, berat_badan, tinggi_badan):
    """
    Transforms user input into a pandas DataFrame that matches the model's training schema.
    """
    # Converting scalar inputs into a DataFrame format.
    # This structure must strictly match the columns used during model training (X_train).
    df = pd.DataFrame({
        "Jenis Kelamin": [gender],
        "Umur (bulan)": [age],
        "Tinggi Badan (cm)": [tinggi_badan], 
        "Berat Badan (kg)": [berat_badan]
    })
    return df
