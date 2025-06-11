from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import joblib

app = FastAPI()

# Load models and scalers
model1 = joblib.load(open("model1.pkl", "rb"))  # Logistic Regression 1
model2 = joblib.load(open("model2.pkl", "rb"))  # XGBoost
model3 = joblib.load(open("model3.pkl", "rb"))  # Logistic Regression 2

scaler1 = joblib.load(open("scaler1.pkl", "rb"))
scaler3 = joblib.load(open("scaler3.pkl", "rb"))


# Define input data model
class ClinicalInput(BaseModel):
    Age: int
    BMI: float
    Insulin: int
    Glucose: float

    High_Blood_Pressure: float
    High_Cholesterol: float
    CholCheck: float
    Smoker: float
    Stroke: float
    cardiovascular_disease: float
    PhysActivity: float
    Fruits: float
    Veggies: float
    HvyAlcoholConsump: float
    AnyHealthcare: float
    NoDocbcCost: float
    DiffWalk: float
    Gender: float
    Education: float
    Income: float

    Urea: float
    Cr: int
    HbA1c: float
    Total_Cholesterol: float
    Triglycerides: float
    LDL: float
    VLDL: float


@app.post("/predict_clinical")
def predict_clinical(data: ClinicalInput):
    input_data = data.dict()

    # Model 1 features
    features_model1 = np.array([[
        input_data['Age'], input_data['BMI'], input_data['Insulin'],
        input_data['Glucose']
    ]])

    # Model 2 features (not scaled)
    features_model2 = np.array([[
        input_data['High_Blood_Pressure'], input_data['High_Cholesterol'], input_data['CholCheck'],
        input_data['BMI'], input_data['Smoker'], input_data['Stroke'],
        input_data['cardiovascular_disease'], input_data['PhysActivity'],
        input_data['Fruits'], input_data['Veggies'], input_data['HvyAlcoholConsump'],
        input_data['AnyHealthcare'], input_data['NoDocbcCost'], input_data['DiffWalk'],
        input_data['Gender'], input_data['Age'], input_data['Education'], input_data['Income']
    ]])

    # Model 3 features
    features_model3 = np.array([[
        input_data['Gender'], input_data['Age'], input_data['Urea'], input_data['Cr'],
        input_data['HbA1c'], input_data['Total_Cholesterol'], input_data['Triglycerides'],
        input_data['LDL'], input_data['VLDL'], input_data['BMI']
    ]])

    # Scale inputs
    scaled1 = scaler1.transform(features_model1)
    scaled3 = scaler3.transform(features_model3)

    # Predict probabilities
    proba1 = model1.predict_proba(scaled1)
    proba2 = model2.predict_proba(features_model2)
    proba3 = model3.predict_proba(scaled3)

    # Soft voting
    avg_proba = (proba1 + proba2 + proba3) / 3
    final_pred = int(np.argmax(avg_proba))
    confidence = float(np.max(avg_proba))

    return {
        "risk": "High" if final_pred == 1 else "Low",
        "predicted_class": final_pred,
        "confidence": round(confidence, 4),
        "soft_vote_probs": avg_proba.tolist()
    }

