import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Get the absolute path to the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading data...")
df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'student_data.csv'))

print("Cleaning data...")
df['passed'] = (df['G3'] >= 10).astype(int)
df_clean = df.drop(columns=['G1', 'G2', 'G3'])
df_clean = pd.get_dummies(df_clean, drop_first=True)

df_clean.to_csv(os.path.join(BASE_DIR, 'data', 'cleaned_student_data.csv'), index=False)
print("Cleaned data saved.")

print("Training model...")
X = df_clean.drop(columns=['passed'])
y = df_clean['passed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Saving model...")
os.makedirs(os.path.join(BASE_DIR, 'app'), exist_ok=True)
joblib.dump(model, os.path.join(BASE_DIR, 'app', 'model.pkl'))
joblib.dump(list(X.columns), os.path.join(BASE_DIR, 'app', 'features.pkl'))

print("Pipeline executed successfully!")
