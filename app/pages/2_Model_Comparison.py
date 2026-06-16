import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.components.styling import get_custom_css

st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

st.title("📈 Model Comparison")

st.markdown("""
This page displays the evaluation metrics for all the models trained in the pipeline.
Use the dropdown below to select models and view their Confusion Matrices and ROC Curves.
""")

st.info("Metrics and plots will be loaded from MLflow or serialized evaluation results in a full deployment.")
