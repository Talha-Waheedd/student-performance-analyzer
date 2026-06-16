import streamlit as st
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.components.styling import get_custom_css

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

st.title("🎓 Student Performance Predictor")
st.markdown("""
Welcome to the Student Performance Prediction Platform! 
This application leverages advanced machine learning to predict student outcomes and identify areas where interventions might be needed.

### Navigation
- **1_Dashboard**: Make real-time predictions and see SHAP explanations.
- **2_Model_Comparison**: Compare the performance metrics of different models.
- **3_Analytics**: Dive deep into feature importance and student clustering.
- **4_Data_Explorer**: Explore the raw data, correlations, and outliers.

Please select a page from the sidebar to get started.
""")

# Initialize session state for predictions
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
