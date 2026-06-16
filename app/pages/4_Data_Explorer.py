import streamlit as st
import pandas as pd
import sys
import os
import plotly.express as px

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.components.styling import get_custom_css
from src.data.data_loader import load_data, load_config

st.set_page_config(page_title="Data Explorer", page_icon="📁", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

@st.cache_data
def get_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    app_config = load_config(os.path.join(base_dir, 'config', 'app_config.yaml'))
    data_path = os.path.join(base_dir, app_config['data']['raw_data_path'])
    return load_data(data_path)

st.title("📁 Data Explorer")

try:
    df = get_data()
    st.dataframe(df, use_container_width=True)
    
    st.subheader("Data Distribution")
    col_to_plot = st.selectbox("Select column to visualize:", df.columns)
    
    if pd.api.types.is_numeric_dtype(df[col_to_plot]):
        fig = px.histogram(df, x=col_to_plot, marginal="box", template="plotly_dark")
    else:
        fig = px.pie(df, names=col_to_plot, template="plotly_dark")
        
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"Failed to load data: {e}")
