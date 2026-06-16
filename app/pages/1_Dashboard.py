import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
import shap
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.components.styling import get_custom_css

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'student_pipeline.pkl')
    if os.path.exists(model_path):
        from src.models.pipeline import StudentPerformancePipeline
        pipeline = StudentPerformancePipeline(config={})
        pipeline.load(model_path)
        return pipeline
    return None

pipeline = load_model()

st.title("📊 Prediction Dashboard")

if not pipeline:
    st.error("Model not found. Please run the training pipeline first.")
    st.stop()

st.sidebar.header("Student Attributes")
# Provide some default inputs based on dataset
age = st.sidebar.slider("Age", 15, 22, 16, help="Student's age")
studytime = st.sidebar.slider("Study Time (1-4)", 1, 4, 2, help="Weekly study time (1: <2 hours, 2: 2-5 hours, 3: 5-10 hours, 4: >10 hours)")
failures = st.sidebar.slider("Past Class Failures", 0, 4, 0)
absences = st.sidebar.slider("Absences", 0, 93, 2)
G1 = st.sidebar.slider("Period 1 Grade (G1)", 0, 20, 10)
G2 = st.sidebar.slider("Period 2 Grade (G2)", 0, 20, 10)

col1, col2 = st.sidebar.columns(2)
schoolsup = col1.selectbox("School Support", ["yes", "no"])
famsup = col2.selectbox("Family Support", ["yes", "no"])
paid = col1.selectbox("Paid Classes", ["yes", "no"])
activities = col2.selectbox("Extracurricular", ["yes", "no"])
nursery = col1.selectbox("Attended Nursery", ["yes", "no"])
higher = col2.selectbox("Wants Higher Ed", ["yes", "no"])
internet = col1.selectbox("Internet Access", ["yes", "no"])
romantic = col2.selectbox("Romantic Rel.", ["yes", "no"])

Medu = st.sidebar.slider("Mother's Education (0-4)", 0, 4, 2)
Fedu = st.sidebar.slider("Father's Education (0-4)", 0, 4, 2)
health = st.sidebar.slider("Health Status (1-5)", 1, 5, 3)
goout = st.sidebar.slider("Going Out (1-5)", 1, 5, 3)
Dalc = st.sidebar.slider("Workday Alcohol (1-5)", 1, 5, 1)
Walc = st.sidebar.slider("Weekend Alcohol (1-5)", 1, 5, 1)

# Create input dataframe
input_data = pd.DataFrame({
    'age': [age], 'studytime': [studytime], 'failures': [failures], 'absences': [absences],
    'G1': [G1], 'G2': [G2], 'schoolsup': [schoolsup], 'famsup': [famsup], 'paid': [paid],
    'activities': [activities], 'nursery': [nursery], 'higher': [higher], 'internet': [internet],
    'romantic': [romantic], 'Medu': [Medu], 'Fedu': [Fedu], 'health': [health], 'goout': [goout],
    'Dalc': [Dalc], 'Walc': [Walc]
})

col_main1, col_main2 = st.columns([1, 2])

with col_main1:
    st.subheader("Prediction Result")
    if st.button("Predict Performance"):
        # Make prediction
        try:
            prob = pipeline.predict_proba(input_data)[0]
            pred = 1 if prob >= 0.5 else 0
            
            st.metric("Pass Probability", f"{prob*100:.1f}%")
            
            if prob >= 0.7:
                st.markdown('<div class="risk-low">Risk Level: Low</div>', unsafe_allow_html=True)
            elif prob >= 0.4:
                st.markdown('<div class="risk-medium">Risk Level: Medium</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-high">Risk Level: High</div>', unsafe_allow_html=True)
                
            st.session_state.prediction_history.append({"prob": prob, "pred": pred})
            
            # SHAP Explanation (simplified for sklearn pipeline)
            # We will use TreeExplainer if it's a tree model, or KernelExplainer
            with col_main2:
                st.subheader("Prediction Explanation")
                st.info("SHAP values will be calculated here in a full deployment. Due to pipeline complexity (custom transformers), a generic feature importance could be shown instead or KernelExplainer used.")
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with col_main2:
    st.subheader("What-If Analysis")
    st.markdown("Adjust the sliders on the left to see how the prediction changes in real-time.")
    
