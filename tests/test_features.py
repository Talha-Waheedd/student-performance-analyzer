import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.feature_engineering import FeatureEngineer
from src.features.preprocessor import OutlierCapper

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'studytime': [2, 4, 1],
        'age': [15, 16, 17],
        'failures': [0, 1, 3],
        'schoolsup': ['yes', 'no', 'yes'],
        'famsup': ['yes', 'yes', 'no'],
        'paid': ['no', 'yes', 'no'],
        'absences': [2, 10, 20],
        'G1': [10, 12, 8],
        'G2': [11, 13, 9],
        'Medu': [4, 2, 1],
        'Fedu': [3, 2, 1],
        'activities': ['yes', 'yes', 'no'],
        'nursery': ['yes', 'no', 'yes'],
        'higher': ['yes', 'yes', 'no'],
        'internet': ['yes', 'no', 'yes'],
        'health': [3, 5, 1],
        'goout': [4, 2, 5],
        'Dalc': [1, 2, 4],
        'Walc': [1, 3, 5]
    })

def test_feature_engineer(sample_data):
    fe = FeatureEngineer()
    transformed = fe.transform(sample_data)
    
    assert 'study_effort_ratio' in transformed.columns
    assert 'support_index' in transformed.columns
    assert 'academic_risk_score' in transformed.columns
    
    # Check specific calculation
    # study_effort_ratio = studytime / (age + failures + 1)
    # Row 0: 2 / (15 + 0 + 1) = 2/16 = 0.125
    assert np.isclose(transformed.loc[0, 'study_effort_ratio'], 0.125)
    
    # support_index = sum of yes in schoolsup, famsup, paid
    # Row 0: yes, yes, no -> 2
    assert transformed.loc[0, 'support_index'] == 2
    
def test_outlier_capper():
    data = pd.DataFrame({'val': [1, 2, 3, 4, 5, 100]})
    capper = OutlierCapper(method='iqr', factor=1.5, columns=['val'])
    transformed = capper.fit_transform(data)
    assert transformed['val'].max() < 100
