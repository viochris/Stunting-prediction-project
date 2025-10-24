import pandas as pd

def preprocess_input(gender, age, berat_badan, tinggi_badan):
    df = pd.DataFrame({
        "Jenis Kelamin": [gender],
        "Umur (bulan)": [age],
        "Tinggi Badan (cm)": [berat_badan],
        "Berat Badan (kg)": [tinggi_badan]
    })
    return df