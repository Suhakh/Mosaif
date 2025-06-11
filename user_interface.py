import streamlit as st
import requests
import pandas as pd
from datetime import datetime

import base64
from PIL import Image
from io import BytesIO



# Set the sidebar navigation
st.set_page_config(layout="wide")
st.sidebar.image("logo.png", use_container_width=True)  # Make sure logo.png is present
st.sidebar.title("MOSAIF")
page = st.sidebar.radio("Go to", ["🧍 Digital Twin", "📊 Analytics Dashboard", "🧬 Prediction Panel"])

# Page: Digital Twin
if page == "🧍 Digital Twin":
    st.title("Digital Twin: 3D Body Representation")
    st.markdown("Build your own digital twin that predicts your T2D risk.")
    st.image("body.png", use_container_width=True)  # Make sure body.png is present

# Page: Analytics Dashboard
elif page == "📊 Analytics Dashboard":
    st.title("📈 T2D Risk Analytics")
    # ... (rest of Analytics Dashboard code remains the same) ...
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧮 Risk Contribution Breakdown")
        if "clinical_prob" in st.session_state and "genetic_prob" in st.session_state:
            risk_data = pd.DataFrame({
                'Source': ['Clinical', 'Genetic'],
                'Probability': [st.session_state.clinical_prob, st.session_state.genetic_prob]
            })
            st.bar_chart(risk_data.set_index("Source"))
        else:
            st.info("Run a prediction to see risk contributions.")

    with col2:
        st.subheader("🕒 Historical Predictions")
        history = st.session_state.get("prediction_history", [])
        if history:
            df_history = pd.DataFrame(history)
            df_history["timestamp"] = pd.to_datetime(df_history["timestamp"])
            df_history = df_history.sort_values("timestamp")
            st.line_chart(df_history.set_index("timestamp")[["fused_score"]])
        else:
            st.info("No prediction history available yet.")

    st.subheader("🧠 Clinical vs Genetic Influence")
    if "clinical_prob" in st.session_state and "genetic_prob" in st.session_state:
        influence_df = pd.DataFrame({
            "Factor": ["Clinical Model", "Genetic Model"],
            "Risk Probability": [st.session_state.clinical_prob, st.session_state.genetic_prob]
        })
        st.bar_chart(influence_df.set_index("Factor"))
    else:
        st.info("Prediction required to compare influence.")


# Page: Prediction Panel
elif page == "🧬 Prediction Panel":
    st.title("🧬 Type 2 Diabetes Risk Prediction")
    st.markdown("Provide your clinical and genetic data to predict your **risk level** and get a **preventive plan**.")

    with st.expander("🩺 Clinical Information", expanded=True):
        # ... (clinical inputs remain the same) ...
        col1, col2, col3 = st.columns(3)
        with col1:
            Age = st.number_input("Age", min_value=18, max_value=120, value=30)  # Min age 18 for plan
            BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, format="%.1f")
            Insulin = st.number_input("Insulin (mu U/ml)", value=85, min_value=0)
            Glucose = st.number_input("Glucose (mg/dL)", value=100.0, min_value=0.0, format="%.1f")
            High_Blood_Pressure = st.radio("High Blood Pressure History", [0.0, 1.0],
                                           format_func=lambda x: "Yes" if x == 1.0 else "No")
            High_Cholesterol = st.radio("High Cholesterol History", [0.0, 1.0],
                                        format_func=lambda x: "Yes" if x == 1.0 else "No")
        with col2:
            CholCheck = st.radio("Cholesterol Check in last 5 years", [0.0, 1.0],
                                 format_func=lambda x: "Yes" if x == 1.0 else "No")
            Smoker = st.radio("Smoker", [0.0, 1.0], format_func=lambda x: "Yes" if x == 1.0 else "No")
            Stroke = st.radio("Stroke History", [0.0, 1.0], format_func=lambda x: "Yes" if x == 1.0 else "No")
            cardiovascular_disease = st.radio("Cardiovascular Disease History", [0.0, 1.0],
                                              format_func=lambda x: "Yes" if x == 1.0 else "No")
            PhysActivity = st.radio("Any Physical Activity in past 30 days", [0.0, 1.0],
                                    format_func=lambda x: "Yes" if x == 1.0 else "No")  # Used by clinical model
            Fruits = st.radio("Consume Fruit 1 or more times per day", [0.0, 1.0],
                              format_func=lambda x: "Yes" if x == 1.0 else "No")
            Veggies = st.radio("Consume Vegetables 1 or more times per day", [0.0, 1.0],
                               format_func=lambda x: "Yes" if x == 1.0 else "No")
        with col3:
            HvyAlcoholConsump = st.radio("Heavy Alcohol Consumption (men >14d/wk, women >7d/wk)", [0.0, 1.0],
                                         format_func=lambda x: "Yes" if x == 1.0 else "No")
            AnyHealthcare = st.radio("Have any kind of health care coverage", [0.0, 1.0],
                                     format_func=lambda x: "Yes" if x == 1.0 else "No")
            NoDocbcCost = st.radio("Could not see doctor due to cost (past 12m)", [0.0, 1.0],
                                   format_func=lambda x: "Yes" if x == 1.0 else "No")
            DiffWalk = st.radio("Serious difficulty walking or climbing stairs", [0.0, 1.0],
                                format_func=lambda x: "Yes" if x == 1.0 else "No")
            Gender = st.radio("Gender", [0.0, 1.0],
                              format_func=lambda x: "Male" if x == 1.0 else "Female")  # 0=Female, 1=Male
            Education = st.selectbox("Education Level (1=No School, 6=College Grad)", list(range(1, 7)), index=3)
            Income = st.selectbox("Income Level (1=<$10k, 8=>$75k)", list(range(1, 9)), index=4)

        st.subheader("🧪 Clinical Test Results (Optional)")
        col4, col5, col6 = st.columns(3)
        with col4:
            Urea = st.number_input("Urea (mg/dL)", value=25.0, min_value=0.0, format="%.1f")
            Cr = st.number_input("Creatinine (mg/dL)", value=1, min_value=0)  # Should be float, e.g. 1.0
        with col5:
            HbA1c = st.number_input("HbA1c (%)", value=5.7, min_value=0.0, format="%.1f")
            Total_Cholesterol = st.number_input("Total Cholesterol (mg/dL)", value=200.0, min_value=0.0, format="%.1f")
        with col6:
            Triglycerides = st.number_input("Triglycerides (mg/dL)", value=150.0, min_value=0.0, format="%.1f")
            LDL = st.number_input("LDL Cholesterol (mg/dL)", value=100.0, min_value=0.0, format="%.1f")
            VLDL = st.number_input("VLDL Cholesterol (mg/dL)", value=30.0, min_value=0.0, format="%.1f")

    with st.expander("🧬 Genetic Information (Optional)", expanded=False):  # Default closed
        # ... (genetic inputs remain the same) ...
        CHR_ID = st.number_input("Chromosome ID", value=1, min_value=0)
        CHR_POS = st.number_input("Chromosome Position", value=123456.0, min_value=0.0, format="%f")
        SNPS = st.number_input("SNPs Count", value=1, min_value=0)
        SNP_ID_CURRENT = st.number_input("SNP ID (Numeric)", value=101.0, min_value=0.0,
                                         format="%f")  # rsIDs are usually alphanumeric
        INTERGENIC = st.number_input("Intergenic Score", value=0.5, min_value=0.0, format="%.1f")
        RISK_ALLELE_FREQUENCY = st.slider("Risk Allele Frequency", 0.0, 1.0, 0.2, format="%.2f")
        P_VALUE = st.number_input("P-Value", value=0.01, min_value=0.0, max_value=1.0, format="%f")
        PVALUE_MLOG = st.number_input("P-Value -log10", value=2.0, min_value=0.0, format="%.1f")
        OR_or_BETA = st.number_input("OR or Beta", value=1.1, min_value=0.0, format="%.2f")
        PRS_scaled = st.number_input("PRS_scaled", value=3.0 , min_value=0.0, format="%.2f")
        Ethnicity = st.selectbox("Ethnicity",
                                 ["European", "African", "Asian"])  # Default to European to match common GWAS

    ethnicity_african = int(Ethnicity == "African")
    ethnicity_asian = int(Ethnicity == "Asian")
    ethnicity_european = int(Ethnicity == "European")

    # New Expander for Lifestyle Information for Preventive Plan
    with st.expander("🏃 Lifestyle Information (for Preventive Plan)", expanded=True):
        activity_level = st.selectbox(
            "Typical Physical Activity Level",
            options=["non_active", "moderately_active", "active"],
            index=1,  # Default to moderately_active
            help="Select your general physical activity level."
        )
        diet_habits = st.selectbox(
            "Predominant Dietary Habits",
            options=["balanced_diet", "processed_food", "high_sugar_intake"],
            index=0,  # Default to balanced_diet
            help="Select the category that best describes your typical diet."
        )

    if st.button("🔍 Predict Risk & Get Plan"):
        with st.spinner("Predicting and generating plan..."):
            # Ensure Cr is int as per Pydantic model in final_prediction.py
            # but it's better to keep it float in UI and cast if necessary in backend
            # For now, assuming UI passes float and Pydantic handles it or backend casts.
            # Actually, ClinicalInput has Cr: int. So UI should send int.
            cr_value = int(Cr) if isinstance(Cr, float) else Cr

            payload = {
                "clinical_data": {
                    "Age": Age, "BMI": BMI, "Insulin": Insulin, "Glucose": Glucose,
                    "High_Blood_Pressure": High_Blood_Pressure, "High_Cholesterol": High_Cholesterol,
                    "CholCheck": CholCheck, "Smoker": Smoker, "Stroke": Stroke,
                    "cardiovascular_disease": cardiovascular_disease, "PhysActivity": PhysActivity,
                    "Fruits": Fruits, "Veggies": Veggies, "HvyAlcoholConsump": HvyAlcoholConsump,
                    "AnyHealthcare": AnyHealthcare, "NoDocbcCost": NoDocbcCost, "DiffWalk": DiffWalk,
                    "Gender": Gender, "Education": float(Education), "Income": float(Income),
                    # Ensure float for Pydantic
                    "Urea": Urea, "Cr": cr_value, "HbA1c": HbA1c,
                    "Total_Cholesterol": Total_Cholesterol, "Triglycerides": Triglycerides,
                    "LDL": LDL, "VLDL": VLDL
                },
                "genetic_data": {
                    "CHR_ID": CHR_ID, "CHR_POS": CHR_POS, "SNPS": SNPS,
                    "SNP_ID_CURRENT": SNP_ID_CURRENT, "INTERGENIC": INTERGENIC,
                    "RISK_ALLELE_FREQUENCY": RISK_ALLELE_FREQUENCY,
                    "P_VALUE": P_VALUE, "PVALUE_MLOG": PVALUE_MLOG,
                    "OR_or_BETA": OR_or_BETA, "PRS_scaled" : PRS_scaled,
                    "Ethnicity_African": ethnicity_african,
                    "Ethnicity_Asian": ethnicity_asian,
                    "Ethnicity_European": ethnicity_european
                },
                "lifestyle_data": {  # Added lifestyle data
                    "activity_level": activity_level,
                    "diet_habits": diet_habits
                }
            }

            try:
                # Ensure your FastAPI app (final_prediction.py) is running on port 8004
                response = requests.post("http://localhost:8004/final_prediction", json=payload)
                response.raise_for_status()  # Raise an exception for HTTP errors

                result = response.json()
                st.success(f"🧾 **Risk Prediction: {result['final_risk']}**")
                st.info(f"🔢 Fused Score: {result['fused_score']}")
                st.info(f"🩺 Clinical Probability: {result['clinical_prob']}")
                st.info(f"🧬 Genetic Probability: {result['genetic_prob']}")

                # Store current values in session_state
                st.session_state.clinical_prob = result["clinical_prob"]
                st.session_state.genetic_prob = result["genetic_prob"]
                st.session_state.fused_score = result["fused_score"]

                # Add to history
                if "prediction_history" not in st.session_state:
                    st.session_state.prediction_history = []
                st.session_state.prediction_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "clinical_prob": result["clinical_prob"],
                    "genetic_prob": result["genetic_prob"],
                    "fused_score": result["fused_score"]
                })

                # Display the preventive plan
                if "preventive_plan" in result:
                    st.subheader("📝 Personalized Preventive Plan")
                    # REMOVE the <pre> tag, just pass the markdown string directly
                    st.markdown(result['preventive_plan'],
                                unsafe_allow_html=True)  # unsafe_allow_html might still be needed if you embed any HTML by mistake, but for pure markdown it's often not. Test without it too.
                else:
                    st.warning("Preventive plan not available in the response.")

            except requests.exceptions.HTTPError as http_err:
                st.error(f"❌ HTTP error occurred: {http_err}")
                st.error(f"Response content: {response.text}")
            except requests.exceptions.ConnectionError as conn_err:
                st.error(f"❌ Connection error: {conn_err}. Ensure the backend API is running at http://localhost:8004.")
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

