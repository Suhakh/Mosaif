
import shap
import matplotlib.pyplot as plt
import os
import uuid
from fastapi.responses import FileResponse

def explain_risk_factors(clinical_data: ClinicalInput):
    d = clinical_data.dict()

    # --- Prepare inputs for all 3 models ---
    f1 = np.array([[d['Age'], d['BMI'], d['Insulin'], d['Glucose']]])
    f2 = np.array([[
        d['High_Blood_Pressure'], d['High_Cholesterol'], d['CholCheck'],
        d['BMI'], d['Smoker'], d['Stroke'], d['cardiovascular_disease'],
        d['PhysActivity'], d['Fruits'], d['Veggies'], d['HvyAlcoholConsump'],
        d['AnyHealthcare'], d['NoDocbcCost'], d['DiffWalk'], d['Gender'],
        d['Age'], d['Education'], d['Income']
    ]])
    f3 = np.array([[d['Gender'], d['Age'], d['Urea'], d['Cr'], d['HbA1c'],
                    d['Total_Cholesterol'], d['Triglycerides'], d['LDL'],
                    d['VLDL'], d['BMI']]])

    # --- Apply scaling ---
    f1_scaled = scaler1.transform(f1)
    f3_scaled = scaler3.transform(f3)

    # --- SHAP Explainers ---
    explainer1 = shap.Explainer(model1)
    explainer2 = shap.Explainer(model2)
    explainer3 = shap.Explainer(model3)

    shap_values1 = explainer1(f1_scaled)
    shap_values2 = explainer2(f2)
    shap_values3 = explainer3(f3_scaled)

    # --- Concatenate SHAP values and features ---
    all_shap_values = np.concatenate([shap_values1.values, shap_values2.values, shap_values3.values], axis=1)
    all_features = np.concatenate([f1_scaled, f2, f3_scaled], axis=1)

    feature_names = (
        ['Age', 'BMI', 'Insulin', 'Glucose'] +
        ['High_BP', 'High_Cholesterol', 'CholCheck', 'BMI2', 'Smoker', 'Stroke', 'CVD', 'PhysAct', 'Fruits', 'Veggies',
         'Alcohol', 'Healthcare', 'NoDocbcCost', 'DiffWalk', 'Gender2', 'Age2', 'Edu', 'Income'] +
        ['Gender3', 'Age3', 'Urea', 'Cr', 'HbA1c', 'Tot_Chol', 'Trigly', 'LDL', 'VLDL', 'BMI3']
    )

    # --- Plot SHAP summary ---
    fig = plt.figure()
    shap.summary_plot(all_shap_values, all_features, feature_names=feature_names, show=False)
    plot_filename = f"shap_plot_{uuid.uuid4().hex}.png"
    plot_path = os.path.join("explanations", plot_filename)
    os.makedirs("explanations", exist_ok=True)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    # --- Explanation Text (top contributors) ---
    mean_abs = np.abs(all_shap_values[0]).mean(axis=0)
    top_indices = np.argsort(-mean_abs)[:3]
    top_features = [feature_names[i] for i in top_indices]
    explanation_text = f"Top contributing features: {', '.join(top_features)}."

    return plot_path, explanation_text

@app.post("/explain_clinical_risk")
def explain_clinical_risk(clinical_data: ClinicalInput):
    plot_path, text = explain_risk_factors(clinical_data)
    return {
        "explanation_text": text,
        "explanation_image_url": f"/get_explanation_image/{os.path.basename(plot_path)}"
    }

@app.get("/get_explanation_image/{filename}")
def get_explanation_image(filename: str):
    file_path = os.path.join("explanations", filename)
    return FileResponse(file_path, media_type="image/png")
