from sklearn.preprocessing import MinMaxScaler
import joblib

def load_model():
    gender_enc = joblib.load("models/Jenis Kelamin_encoder.joblib")
    stunting_enc = joblib.load("models/Stunting_encoder.joblib")
    scaler = joblib.load("models/scaler.joblib")
    best_model = joblib.load("models/best_model.joblib")
    
    return gender_enc, stunting_enc, scaler, best_model

def predict(gender_enc, stunting_enc, scaler, best_model, input_data):
    df = input_data
    df["Jenis Kelamin"] = gender_enc.transform(df["Jenis Kelamin"])
    
    cols = df.drop("Jenis Kelamin", axis = 1).columns
    df[cols] = scaler.transform(df[cols])
    
    prediction = best_model.predict(df)
    result = stunting_enc.inverse_transform(prediction)[0]
    return result