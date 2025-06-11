# Mosaif


# MOSAIF: Personalized Type 2 Diabetes Risk Prediction and Preventive Planning System

MOSAIF is a full-stack, AI-powered digital health platform designed to predict an individual’s risk of developing Type 2 Diabetes (T2D) by fusing clinical, genetic, and lifestyle data. It also provides a personalized, expert-validated 7-day preventive plan to reduce the user’s risk based on their unique profile. The platform aims to empower individuals, clinicians, and researchers with actionable and explainable health intelligence.

---

## 🧠 Key Features

- **T2D Risk Prediction** using a fusion of clinical and genetic datasets
- **Explainable AI** via SHAP to ensure transparency
- **Personalized 7-Day Plan Generator** (diet + workout) based on lifestyle & risk drivers
- **Web Interface** using Streamlit for user-friendly interaction
- **FastAPI Backend** for real-time predictions and plan generation

---

## 📈 Machine Learning Pipeline

### Clinical Model
- Algorithms: Logistic Regression, XGBoost
- Ensemble: Soft Voting Classifier
- Preprocessing: Encoding, normalization, SMOTE

### Genetic Model
- Dataset: SNPs filtered for T2D
- Method: Logistic Regression on binarized gPRS
- Preprocessing: Log transformation, standardization, Calculating gPRS

### Fusion Strategy
- Weighted Average Combination based on AUC scores:
  
  \[
  P_{\text{final}} = w_{clinical} \cdot P_{clinical} + w_{genetic} \cdot P_{genetic}
  \]

---

## 🥗 Personalized Plan Generator

- Rule-based system accessing JSON files:
  - `high_diet_plans.json`, `low_diet_plans.json`
  - `high_workout_plans.json`, `low_workout_plans.json`
- Based on: Gender, Age Group, Activity Level, Dietary Habit, Primary Risk Factor
- Output: Custom 7-day schedule of meals + exercise routines

---

## 💻 Technologies Used

- **Frontend**: Streamlit
- **Backend**: FastAPI + Pydantic
- **ML Models**: scikit-learn, XGBoost
- **Explainability**: SHAP
- **Storage & Logic**: JSON Knowledge Base
- **Data Processing**: pandas, numpy
- **Deployment Ready**: Render, Azure (Free tier compatible)

---

## ⚙️ Installation & Usage


```bash
git clone https://github.com/Suhakh/mosaif.git
cd mosaif

pip install -r requirements.txt

uvicorn final_prediction.py:app --reload --port 8001

streamlit run user_interface.py

