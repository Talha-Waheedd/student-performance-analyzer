import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Page config
st.set_page_config(page_title="Student Performance Analyser", layout="wide", page_icon="🎓")

# Custom CSS for a better look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 Student Performance Analyser")
st.markdown("Predict whether a student will pass or fail based on demographic and academic data.")

import os

# Get the absolute path to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load models and data
@st.cache_resource
def load_resources():
    model = joblib.load(os.path.join(BASE_DIR, 'app', 'model.pkl'))
    features = joblib.load(os.path.join(BASE_DIR, 'app', 'features.pkl'))
    df_raw = pd.read_csv(os.path.join(BASE_DIR, 'data', 'student_data.csv'))
    return model, features, df_raw

try:
    model, features, df_raw = load_resources()
except Exception as e:
    st.error("Error loading models or data. Please ensure the pipeline has been executed.")
    st.stop()

# Sidebar for user inputs
st.sidebar.header("Student Profile Input")
st.sidebar.markdown("Enter student details to predict their performance.")

# Dynamically generate inputs based on original dataset (excluding grades)
input_data = {}
cols_to_drop = ['G1', 'G2', 'G3']

for col in df_raw.columns:
    if col in cols_to_drop:
        continue
    
    if pd.api.types.is_numeric_dtype(df_raw[col]):
        min_val = int(df_raw[col].min())
        max_val = int(df_raw[col].max())
        mean_val = int(df_raw[col].mean())
        # Avoid issues where min == max
        if min_val == max_val:
            input_data[col] = st.sidebar.number_input(f"{col.capitalize()}", value=min_val)
        else:
            input_data[col] = st.sidebar.slider(f"{col.capitalize()}", min_val, max_val, mean_val)
    else:
        options = df_raw[col].unique().tolist()
        input_data[col] = st.sidebar.selectbox(f"{col.capitalize()}", options)

# Prediction Button
if st.sidebar.button("Predict Performance"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Preprocess
    input_encoded = pd.get_dummies(input_df)
    
    # Align columns with model features
    input_aligned = input_encoded.reindex(columns=features, fill_value=0)
    
    # Predict
    prediction = model.predict(input_aligned)[0]
    prob = model.predict_proba(input_aligned)[0]
    
    st.markdown("---")
    st.header("Prediction Result")
    
    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.success("🎉 This student is predicted to **PASS**.")
        else:
            st.error("⚠️ This student is predicted to **FAIL**.")
            
    with col2:
        st.info(f"Confidence Score: **{max(prob)*100:.1f}%**")
        
    # Show feature importance chart
    st.markdown("### Feature Importance")
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:] # Top 10
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(range(len(indices)), importances[indices], align='center', color='#3498db')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([features[i] for i in indices])
    ax.set_title('Top 10 Important Features for Prediction')
    st.pyplot(fig)

st.markdown("---")
st.header("Exploratory Data Analysis")
st.markdown("Here is a quick look at the dataset distributions.")

tab1, tab2 = st.tabs(["Score Distribution", "Absences vs Pass"])

with tab1:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df_raw['G3'], bins=20, kde=True, color='skyblue', ax=ax)
    ax.set_title('Distribution of Final Grades (G3)')
    ax.axvline(10, color='red', linestyle='--', label='Passing Threshold (10)')
    ax.legend()
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(8, 4))
    # Pass status for visualization
    df_raw['passed'] = (df_raw['G3'] >= 10).map({True: 'Pass', False: 'Fail'})
    sns.boxplot(x='passed', y='absences', data=df_raw, palette='pastel', ax=ax)
    ax.set_title('Absences vs Pass/Fail Status')
    st.pyplot(fig)
