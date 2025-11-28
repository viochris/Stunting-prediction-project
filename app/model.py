from sklearn.preprocessing import MinMaxScaler
import joblib

def load_model():
    """
    Loads serialized model artifacts from the disk into memory.
    """
    # ---------------------------------------------------------
    # ARTIFACT DESERIALIZATION (LOADING FROZEN ASSETS)
    # ---------------------------------------------------------
    # We are not training the model here. We are loading the "state" of the model
    # that was saved after the training phase. This ensures reproducibility.
    
    # 1. Encoders: To translate categorical text (e.g., 'Laki-laki') into numbers the model understands.
    gender_enc = joblib.load("models/Jenis Kelamin_encoder.joblib")
    stunting_enc = joblib.load("models/Stunting_encoder.joblib")
    
    # 2. Scaler: Crucial for distance-based models (like KNN) or gradient-based models.
    # It ensures the new input data has the same distribution (min/max) as the training data.
    scaler = joblib.load("models/scaler.joblib")
    
    # 3. Classifier: The trained algorithm (Random Forest/XGBoost/etc.) ready for inference.
    best_model = joblib.load("models/best_model.joblib")
    
    return gender_enc, stunting_enc, scaler, best_model

def predict(gender_enc, stunting_enc, scaler, best_model, input_data):
    """
    The main Inference Pipeline: Preprocessing -> Prediction -> Postprocessing.
    """
    df = input_data
    
    # ---------------------------------------------------------
    # STEP 1: CATEGORICAL ENCODING
    # ---------------------------------------------------------
    # The model cannot understand strings like 'Perempuan'. 
    # We use the fitted encoder to transform it into the specific integer learned during training.
    df["Jenis Kelamin"] = gender_enc.transform(df["Jenis Kelamin"])
    
    # ---------------------------------------------------------
    # STEP 2: FEATURE SCALING (NORMALIZATION)
    # ---------------------------------------------------------
    # Raw numbers (e.g., Age=20, Height=90) have different scales.
    # We apply the SAVED scaler stats to normalize these values between 0 and 1.
    # NOTE: We exclude 'Jenis Kelamin' because it's already categorical/encoded.
    cols = df.drop("Jenis Kelamin", axis = 1).columns
    df[cols] = scaler.transform(df[cols])
    
    # ---------------------------------------------------------
    # STEP 3: MODEL INFERENCE
    # ---------------------------------------------------------
    # The mathematical calculation happens here based on the weights learned by the model.
    # Output is an array of class indices (e.g., [0] or [1]).
    prediction = best_model.predict(df)
    
    # ---------------------------------------------------------
    # STEP 4: DECODING (INVERSE TRANSFORM)
    # ---------------------------------------------------------
    # Converting the mathematical output (integer) back to a human-readable label
    # (e.g., 0 -> 'Severe Stunting', 1 -> 'Normal').
    result = stunting_enc.inverse_transform(prediction)[0]
    
    return result
