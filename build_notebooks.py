import json

def create_notebook(cells, filename):
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    for ctype, content in cells:
        if ctype == 'markdown':
            cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\\n" for line in content.split("\\n")]
            }
        else:
            cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\\n" for line in content.split("\\n")]
            }
        notebook["cells"].append(cell)
        
    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=2)

# 01 Data Cleaning
cells1 = [
    ('markdown', "# Phase 3: Data Cleaning\\nLoading and cleaning the student performance dataset."),
    ('code', """import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('../data/student_data.csv')
df.head()"""),
    ('code', """# Check for missing values
missing_values = df.isnull().sum()
missing_values[missing_values > 0]"""),
    ('markdown', "No missing values or we can handle them if they exist."),
    ('code', """# Create target variable 'passed' based on final grade G3 (>= 10)
df['passed'] = (df['G3'] >= 10).astype(int)

# Drop G1, G2, G3 to prevent data leakage for predicting pass/fail
df_clean = df.drop(columns=['G1', 'G2', 'G3'])

# Encode categorical variables
df_clean = pd.get_dummies(df_clean, drop_first=True)
df_clean.head()"""),
    ('code', """# Save the cleaned dataset
df_clean.to_csv('../data/cleaned_student_data.csv', index=False)
print('Cleaned dataset saved!')""")
]
create_notebook(cells1, 'notebooks/01_data_cleaning.ipynb')

# 02 EDA
cells2 = [
    ('markdown', "# Phase 4: Exploratory Data Analysis (EDA)\\nVisualizing the cleaned data."),
    ('code', """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('../data/student_data.csv')
df_clean = pd.read_csv('../data/cleaned_student_data.csv')

# Set style
sns.set_theme(style="whitegrid")"""),
    ('code', """# Distribution of final grades (G3)
plt.figure(figsize=(8, 5))
sns.histplot(df['G3'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Final Grades (G3)')
plt.xlabel('Final Grade')
plt.ylabel('Count')
plt.show()"""),
    ('code', """# Pass/Fail Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='passed', data=df_clean, palette='pastel')
plt.title('Pass vs Fail Distribution')
plt.xticks([0, 1], ['Fail (<10)', 'Pass (>=10)'])
plt.show()"""),
    ('code', """# Correlation Heatmap
plt.figure(figsize=(12, 10))
# compute correlation only for numeric columns
corr = df_clean.corr()
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()""")
]
create_notebook(cells2, 'notebooks/02_eda_visualisation.ipynb')

# 03 Model Training
cells3 = [
    ('markdown', "# Phase 5: Model Training\\nTraining a classifier to predict pass/fail."),
    ('code', """import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df_clean = pd.read_csv('../data/cleaned_student_data.csv')

X = df_clean.drop(columns=['passed'])
y = df_clean['passed']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"""),
    ('code', """# Train Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)"""),
    ('code', """# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()"""),
    ('code', """# Export the model
joblib.dump(model, '../app/model.pkl')
# Export the feature names so streamlit can know the exact columns needed
joblib.dump(list(X.columns), '../app/features.pkl')
print("Model saved to app/model.pkl")""")
]
create_notebook(cells3, 'notebooks/03_model_training.ipynb')

print("Notebooks created successfully.")
