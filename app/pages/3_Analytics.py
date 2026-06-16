import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.components.styling import get_custom_css

st.set_page_config(page_title="Advanced Analytics", page_icon="🔍", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

st.title("🔍 Advanced Analytics")

st.markdown("""
### Feature Importance
View the most influential features driving student performance predictions.
""")

st.info("Feature importance plots (Permutation/SHAP) and clustering visuals will be rendered here.")

if 'prediction_history' in st.session_state and len(st.session_state.prediction_history) > 0:
    st.subheader("Prediction History Tracking")
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df)
else:
    st.write("No predictions made in this session yet.")
