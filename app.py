import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page layout
st.set_page_config(page_title="Smart Nano-Particle Classifier", layout="centered")

st.title("🧬 AI-Powered Smart Classifier")
st.subheader("Nano-Particle Biodistribution Classifier")
st.write("Predict if nano-particles will achieve High Tumor Retention and optimal Selectivity.")

# Load the pre-trained full Pipeline
@st.cache_resource
def load_model():
    return joblib.load("xgboost_nano_classifier_model.pkl")

try:
    pipeline = load_model()
    
    # Safely extract categorical features info
    preprocessor = pipeline.named_steps["prep"]
    encoder = preprocessor.named_transformers_["cat"]
    
    categorical_features = [
        "NP_Class", "INPs_Core", "Shape", "Size_Category", 
        "Zeta_Category", "Organ or tissue", "HAS_PEG", "Shell Type", "Tumor Site"
    ]
    
    # Build options list dynamically from the encoder
    categories_dict = {col: list(categories) for col, categories in zip(categorical_features, encoder.categories_)}

    st.markdown("---")
    st.subheader("📋 Input Nano-Particle Features")
    
    col1, col2 = st.columns(2)
    input_data = {}
    
    with col1:
        input_data["NP_Class"] = st.selectbox("NP Class", categories_dict["NP_Class"])
        input_data["INPs_Core"] = st.selectbox("INPs Core", categories_dict["INPs_Core"])
        input_data["Shape"] = st.selectbox("Shape", categories_dict["Shape"])
        input_data["Size_Category"] = st.selectbox("Size Category", categories_dict["Size_Category"])
        input_data["Zeta_Category"] = st.selectbox("Zeta Category", categories_dict["Zeta_Category"])
        input_data["Size (nm)"] = st.number_input("Size (nm)", min_value=0.0, value=50.0, step=1.0)

    with col2:
        input_data["Organ or tissue"] = st.selectbox("Organ or Tissue", categories_dict["Organ or tissue"])
        input_data["HAS_PEG"] = st.selectbox("Has PEG?", categories_dict["HAS_PEG"])
        input_data["Shell Type"] = st.selectbox("Shell Type", categories_dict["Shell Type"])
        input_data["Tumor Site"] = st.selectbox("Tumor Site", categories_dict["Tumor Site"])
        input_data["Zeta Potential (mv)"] = st.number_input("Zeta Potential (mv)", value=0.0, step=0.5)
        input_data["Administration Dosages (mg/kg)"] = st.number_input("Dosage (mg/kg)", min_value=0.0, value=5.0, step=0.1)
        input_data["Time point (h)"] = st.number_input("Time Point (h)", min_value=0.0, value=24.0, step=0.5)

    # Order dataframe features exactly as expected by the pipeline
    ordered_features = categorical_features + ["Size (nm)", "Zeta Potential (mv)", "Administration Dosages (mg/kg)", "Time point (h)"]
    input_df = pd.DataFrame([input_data])[ordered_features]

    st.markdown("---")
    if st.button("🔮 Run Prediction", type="primary"):
        predictions = pipeline.predict(input_df)
        probabilities = pipeline.predict_proba(input_df)
        
        # Format outputs
        tumor_pred = "High Retention (>= Median)" if predictions[0][0] == 1 else "Low Retention (< Median)"
        selectivity_pred = "High Selectivity (>= Median)" if predictions[0][1] == 1 else "Low Selectivity (< Median)"
        
        tumor_prob = probabilities[0][1][1] * 100 
        selectivity_prob = probabilities[1][1][1] * 100 
        
        # Display results with metrics and progress bars
        st.subheader("📊 Prediction Results")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(label="Tumor Retention Class", value=tumor_pred)
            st.progress(int(tumor_prob))
            st.caption(f"Confidence: {tumor_prob:.2f}%")
            
        with res_col2:
            st.metric(label="Selectivity Class", value=selectivity_pred)
            st.progress(int(selectivity_prob))
            st.caption(f"Confidence: {selectivity_prob:.2f}%")

except FileNotFoundError:
    st.error("⚠️ Model file 'xgboost_nano_classifier_model.pkl' not found. Please run the training script first.")
except Exception as e:
    st.error(f"⚠️ An unexpected error occurred: {e}")
