from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from preventive_plan_generator import generate_plan
import pandas as pd


import matplotlib.pyplot as plt
import base64
from io import BytesIO
import shap


app = FastAPI(debug=True)

# --- (Load models and scalers - unchanged) ---
model1 = joblib.load("model1.pkl")
model2 = joblib.load("model2.pkl")
model3 = joblib.load("model3.pkl")
scaler1 = joblib.load("scaler1.pkl")
scaler3 = joblib.load("scaler3.pkl")
#genetic_model = joblib.load("final_genetic_model_18_may.pkl")
try:
    genetic_model = joblib.load("final_genetic_model_18_may.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# --- (Pydantic Models - ClinicalInput, GeneticInput, UserLifestyleInput, FusionInput - unchanged) ---
class ClinicalInput(BaseModel):
    Age: int
    BMI: float
    Insulin: int  # This is the one we'll use for "High Insulin Levels"
    Glucose: float  # This is the one we'll use for "High Glucose Level"
    High_Blood_Pressure: float  # This is a binary flag (0 or 1) from user input
    # We need actual BP values for the extract_reasons logic
    # OR we adjust extract_reasons to use this flag.
    # For now, I'll assume we want to use the flag.
    High_Cholesterol: float  # Also a binary flag.
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
    Total_Cholesterol: float  # This is a value we can use for "High Cholesterol" reason
    Triglycerides: float
    LDL: float
    VLDL: float



class GeneticInput(BaseModel):
    CHR_ID: int
    CHR_POS: float
    SNPS: int
    SNP_ID_CURRENT: float
    INTERGENIC: float
    RISK_ALLELE_FREQUENCY: float
    P_VALUE: float
    PVALUE_MLOG: float
    OR_or_BETA: float
    PRS_scaled: float
    Ethnicity_African: int
    Ethnicity_Asian: int
    Ethnicity_European: int


class UserLifestyleInput(BaseModel):
    activity_level: str
    diet_habits: str


class FusionInput(BaseModel):
    clinical_data: ClinicalInput
    genetic_data: GeneticInput
    lifestyle_data: UserLifestyleInput


# --- (predict_clinical_prob, predict_genetic_prob - unchanged) ---
def predict_clinical_prob(data: ClinicalInput) -> float:
    d = data.dict()
    f1 = np.array([[d['Age'], d['BMI'], d['Insulin'], d['Glucose']]])
    scaled1 = scaler1.transform(f1)
    proba1 = model1.predict_proba(scaled1)
    f2 = np.array([[
        d['High_Blood_Pressure'], d['High_Cholesterol'], d['CholCheck'],
        d['BMI'], d['Smoker'], d['Stroke'], d['cardiovascular_disease'],
        d['PhysActivity'], d['Fruits'], d['Veggies'], d['HvyAlcoholConsump'],
        d['AnyHealthcare'], d['NoDocbcCost'], d['DiffWalk'], d['Gender'],
        d['Age'], d['Education'], d['Income']
    ]])
    proba2 = model2.predict_proba(f2)
    f3 = np.array([[d['Gender'], d['Age'], d['Urea'], d['Cr'], d['HbA1c'],
                    d['Total_Cholesterol'], d['Triglycerides'], d['LDL'],
                    d['VLDL'], d['BMI']]])
    scaled3 = scaler3.transform(f3)
    proba3 = model3.predict_proba(scaled3)
    avg_proba = (proba1 + proba2 + proba3) / 3
    return avg_proba[0][1]


"""genetic_features_order = [
    "CHR_ID",
    "CHR_POS",
    "SNPS",
    "SNP_ID_CURRENT",
    "INTERGENIC",
    "RISK_ALLELE_FREQUENCY",
    "P_VALUE",
    "PVALUE_MLOG",
    "OR_or_BETA",
    "PRS_scaled",
    "Ethnicity_African",
    "Ethnicity_Asian",
    "Ethnicity_European"
]"""

def predict_genetic_prob(data: GeneticInput) -> float:
    df = pd.DataFrame([data.dict()])
    return genetic_model.predict_proba(df)[0][1]

"""def predict_genetic_prob(data: GeneticInput) -> float:
    f = np.array([[getattr(data, field) for field in data.__fields__]])
    return genetic_model.predict_proba(f)[0][1]
"""

# --- NEW FUNCTION TO EXTRACT REASONS ---
def extract_potential_risk_factors(clinical_data: ClinicalInput) -> list[str]:
    """
    Extracts human-readable potential risk factors based on clinical data thresholds.
    Note: Thresholds used here are examples and should be clinically validated.
    """
    reasons = []

    # BMI
    if clinical_data.BMI > 30:
        reasons.append("High BMI")
    elif clinical_data.BMI >= 25:  # Optional: Overweight
        reasons.append("Overweight")

    # Glucose - using the 'Glucose' field from ClinicalInput
    if clinical_data.Glucose > 125:  # Example: Fasting glucose for diabetes
        reasons.append("Glucose Levels")  # Matches key in preventive_plan_generator
    elif clinical_data.Glucose > 100:  # Example: Prediabetes
        reasons.append("Elevated Glucose")

    # Blood Pressure - Using the binary flag 'High_Blood_Pressure'
    # Your provided logic used systolic_bp and diastolic_bp which are not in ClinicalInput.
    # So, we adapt to use the available flag.
    if clinical_data.High_Blood_Pressure == 1.0:
        reasons.append("High BP")  # Matches key in preventive_plan_generator

    # Cholesterol - Using the 'Total_Cholesterol' field from ClinicalInput
    # Your provided logic used 'cholesterol' which might be ambiguous.
    # We can also use the binary flag 'High_Cholesterol' if the specific value isn't as important for reasoning here.
    if clinical_data.Total_Cholesterol > 200:  # Example threshold
        reasons.append("Cholesterol")  # Matches key in preventive_plan_generator
    elif clinical_data.High_Cholesterol == 1.0 and "Cholesterol" not in reasons:  # If user flagged high cholesterol
        reasons.append("Cholesterol")

    # Insulin - using the 'Insulin' field from ClinicalInput
    if clinical_data.Insulin > 25:  # Example threshold for hyperinsulinemia / insulin resistance
        reasons.append("Insulin Resistance")  # Matches key in preventive_plan_generator

    # You can add more reasons based on other fields in ClinicalInput if needed
    # e.g., Smoker, Stroke history, etc. These won't map to specific *plans*
    # but can be shown in the "Identified Primary Risk Factors" display and for tips.
    if clinical_data.Smoker == 1.0:
        reasons.append("Smoker")
    if clinical_data.Stroke == 1.0:
        reasons.append("Stroke History")
    if clinical_data.PhysActivity == 0.0:  # No physical activity
        reasons.append("Low Physical Activity")

    # If no specific factors are met, but risk is high, we might add a general one later
    # For now, if reasons is empty, the plan generator handles it.

    return reasons


@app.post("/final_prediction")
def final_prediction(
        inputs: FusionInput,
        weight_clinical: float = 0.8,
        weight_genetic: float = 0.3
):
    clinical_prob = predict_clinical_prob(inputs.clinical_data)
    genetic_prob = predict_genetic_prob(inputs.genetic_data)
    final_score = (weight_clinical * clinical_prob) + (weight_genetic * genetic_prob)
    # Consider capping final_score if it can exceed 1.0 (e.g., final_score = min(final_score, 1.0))
    label = "High" if final_score > 0.5 else "Low"

    risk_factors_for_plan = []  # Initialize
    if label == "High":
        # --- CALL THE NEW FUNCTION TO GET REASONS ---
        risk_factors_for_plan = extract_potential_risk_factors(inputs.clinical_data)
        if not risk_factors_for_plan:  # If no specific factors extracted but risk is high
            risk_factors_for_plan.append("Other High Risk Factors")  # Generic placeholder

    # --- Generate Preventive Plan ---
    risk_level_for_plan = label
    gender_for_plan = "male" if inputs.clinical_data.Gender == 1.0 else "female"
    age_for_plan = inputs.clinical_data.Age
    activity_level_for_plan = inputs.lifestyle_data.activity_level
    diet_habits_for_plan = inputs.lifestyle_data.diet_habits

    preventive_plan_str = generate_plan(
        risk_level=risk_level_for_plan,
        gender=gender_for_plan,
        age=age_for_plan,
        risk_factors_list=risk_factors_for_plan,  # PASS THE EXTRACTED REASONS
        activity_level=activity_level_for_plan,
        diet_habits=diet_habits_for_plan
    )

    return {
        "final_risk": label,
        "fused_score": round(final_score, 4),
        "clinical_prob": round(clinical_prob, 4),
        "genetic_prob": round(genetic_prob, 4),
        "primary_risk_factors": risk_factors_for_plan,
        "preventive_plan": preventive_plan_str
    }

"""app = FastAPI()

# --- (Load models and scalers - unchanged) ---
model1 = joblib.load("model1.pkl")
model2 = joblib.load("model2.pkl")
model3 = joblib.load("model3.pkl")
scaler1 = joblib.load("scaler1.pkl")
scaler3 = joblib.load("scaler3.pkl")
genetic_model = joblib.load("GENETIC_MODEL_EDITED.pkl")


# --- (Pydantic Models - ClinicalInput, GeneticInput, UserLifestyleInput, FusionInput - unchanged) ---
class ClinicalInput(BaseModel):
    Age: int
    BMI: float
    Insulin: int  # This is the one we'll use for "High Insulin Levels"
    Glucose: float  # This is the one we'll use for "High Glucose Level"
    High_Blood_Pressure: float  # This is a binary flag (0 or 1) from user input
    # We need actual BP values for the extract_reasons logic
    # OR we adjust extract_reasons to use this flag.
    # For now, I'll assume we want to use the flag.
    High_Cholesterol: float  # Also a binary flag.
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
    Total_Cholesterol: float  # This is a value we can use for "High Cholesterol" reason
    Triglycerides: float
    LDL: float
    VLDL: float


class GeneticInput(BaseModel):
    # ... (as before)
    CHR_ID: int
    CHR_POS: float
    SNPS: int
    SNP_ID_CURRENT: float
    INTERGENIC: float
    RISK_ALLELE_FREQUENCY: float
    P_VALUE: float
    PVALUE_MLOG: float
    OR_or_BETA: float
    Ethnicity_African: int
    Ethnicity_Asian: int
    Ethnicity_European: int


class UserLifestyleInput(BaseModel):
    activity_level: str
    diet_habits: str


class FusionInput(BaseModel):
    clinical_data: ClinicalInput
    genetic_data: GeneticInput
    lifestyle_data: UserLifestyleInput


# --- (predict_clinical_prob, predict_genetic_prob - unchanged) ---
def predict_clinical_prob(data: ClinicalInput) -> float:
    d = data.dict()
    f1 = np.array([[d['Age'], d['BMI'], d['Insulin'], d['Glucose']]])
    scaled1 = scaler1.transform(f1)
    proba1 = model1.predict_proba(scaled1)
    f2 = np.array([[
        d['High_Blood_Pressure'], d['High_Cholesterol'], d['CholCheck'],
        d['BMI'], d['Smoker'], d['Stroke'], d['cardiovascular_disease'],
        d['PhysActivity'], d['Fruits'], d['Veggies'], d['HvyAlcoholConsump'],
        d['AnyHealthcare'], d['NoDocbcCost'], d['DiffWalk'], d['Gender'],
        d['Age'], d['Education'], d['Income']
    ]])
    proba2 = model2.predict_proba(f2)
    f3 = np.array([[d['Gender'], d['Age'], d['Urea'], d['Cr'], d['HbA1c'],
                    d['Total_Cholesterol'], d['Triglycerides'], d['LDL'],
                    d['VLDL'], d['BMI']]])
    scaled3 = scaler3.transform(f3)
    proba3 = model3.predict_proba(scaled3)
    avg_proba = (proba1 + proba2 + proba3) / 3
    return avg_proba[0][1]


def predict_genetic_prob(data: GeneticInput) -> float:
    f = np.array([[getattr(data, field) for field in data.__fields__]])
    return genetic_model.predict_proba(f)[0][1]


# --- NEW FUNCTION TO EXTRACT REASONS ---
def extract_potential_risk_factors(clinical_data: ClinicalInput) -> list[str]:

#    Extracts human-readable potential risk factors based on clinical data thresholds.
#    Note: Thresholds used here are examples and should be clinically validated.

    reasons = []

    # BMI
    if clinical_data.BMI > 30:
        reasons.append("High BMI")
    elif clinical_data.BMI >= 25:  # Optional: Overweight
        reasons.append("Overweight")

    # Glucose - using the 'Glucose' field from ClinicalInput
    if clinical_data.Glucose > 125:  # Example: Fasting glucose for diabetes
        reasons.append("Glucose Levels")  # Matches key in preventive_plan_generator
    elif clinical_data.Glucose > 100:  # Example: Prediabetes
        reasons.append("Elevated Glucose")

    # Blood Pressure - Using the binary flag 'High_Blood_Pressure'
    # Your provided logic used systolic_bp and diastolic_bp which are not in ClinicalInput.
    # So, we adapt to use the available flag.
    if clinical_data.High_Blood_Pressure == 1.0:
        reasons.append("High BP")  # Matches key in preventive_plan_generator

    # Cholesterol - Using the 'Total_Cholesterol' field from ClinicalInput
    # Your provided logic used 'cholesterol' which might be ambiguous.
    # We can also use the binary flag 'High_Cholesterol' if the specific value isn't as important for reasoning here.
    if clinical_data.Total_Cholesterol > 200:  # Example threshold
        reasons.append("Cholesterol")  # Matches key in preventive_plan_generator
    elif clinical_data.High_Cholesterol == 1.0 and "Cholesterol" not in reasons:  # If user flagged high cholesterol
        reasons.append("Cholesterol")

    # Insulin - using the 'Insulin' field from ClinicalInput
    if clinical_data.Insulin > 25:  # Example threshold for hyperinsulinemia / insulin resistance
        reasons.append("Insulin Resistance")  # Matches key in preventive_plan_generator

    # You can add more reasons based on other fields in ClinicalInput if needed
    # e.g., Smoker, Stroke history, etc. These won't map to specific *plans*
    # but can be shown in the "Identified Primary Risk Factors" display and for tips.
    if clinical_data.Smoker == 1.0:
        reasons.append("Smoker")
    if clinical_data.Stroke == 1.0:
        reasons.append("Stroke History")
    if clinical_data.PhysActivity == 0.0:  # No physical activity
        reasons.append("Low Physical Activity")

    # If no specific factors are met, but risk is high, we might add a general one later
    # For now, if reasons is empty, the plan generator handles it.

    return reasons


def get_shap_plot_base64(model, scaler, feature_array, feature_names):
    explainer = shap.Explainer(model, feature_array)
    shap_values = explainer(feature_array)

    # Plot SHAP explanation
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.read()).decode('utf-8')


@app.post("/final_prediction")
def final_prediction(
        inputs: FusionInput,
        weight_clinical: float = 0.8,
        weight_genetic: float = 0.3
):
    clinical_prob = predict_clinical_prob(inputs.clinical_data)
    genetic_prob = predict_genetic_prob(inputs.genetic_data)
    final_score = (weight_clinical * clinical_prob) + (weight_genetic * genetic_prob)
    # Consider capping final_score if it can exceed 1.0 (e.g., final_score = min(final_score, 1.0))
    label = "High" if final_score > 0.5 else "Low"

    risk_factors_for_plan = []  # Initialize
    if label == "High":
        # --- CALL THE NEW FUNCTION TO GET REASONS ---
        risk_factors_for_plan = extract_potential_risk_factors(inputs.clinical_data)
        if not risk_factors_for_plan:  # If no specific factors extracted but risk is high
            risk_factors_for_plan.append("Other High Risk Factors")  # Generic placeholder

    # --- Generate Preventive Plan ---
    risk_level_for_plan = label
    gender_for_plan = "male" if inputs.clinical_data.Gender == 1.0 else "female"
    age_for_plan = inputs.clinical_data.Age
    activity_level_for_plan = inputs.lifestyle_data.activity_level
    diet_habits_for_plan = inputs.lifestyle_data.diet_habits

    preventive_plan_str = generate_plan(
        risk_level=risk_level_for_plan,
        gender=gender_for_plan,
        age=age_for_plan,
        risk_factors_list=risk_factors_for_plan,  # PASS THE EXTRACTED REASONS
        activity_level=activity_level_for_plan,
        diet_habits=diet_habits_for_plan
    )
    # Prepare SHAP for model1 as an example
    clinical_dict = inputs.clinical_data.dict()
    f1 = np.array([[clinical_dict['Age'], clinical_dict['BMI'], clinical_dict['Insulin'], clinical_dict['Glucose']]])
    scaled1 = scaler1.transform(f1)
    shap_img_base64 = get_shap_plot_base64(model1, scaler1, scaled1, ["Age", "BMI", "Insulin", "Glucose"])

    return {
        "final_risk": label,
        "fused_score": round(final_score, 4),
        "clinical_prob": round(clinical_prob, 4),
        "genetic_prob": round(genetic_prob, 4),
        "preventive_plan": preventive_plan_str,
        "shap_image": shap_img_base64  # NEW
    }"""