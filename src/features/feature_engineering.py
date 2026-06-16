import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer to add new derived features.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # 1. study_effort_ratio
        if all(col in X_out.columns for col in ['studytime', 'age', 'failures']):
            X_out['study_effort_ratio'] = X_out['studytime'] / (X_out['age'] + X_out['failures'] + 1)
            
        # 2. support_index
        support_cols = ['schoolsup', 'famsup', 'paid']
        if all(col in X_out.columns for col in support_cols):
            X_out['support_index'] = sum((X_out[col] == 'yes').astype(int) for col in support_cols)
            
        # 3. academic_risk_score
        risk_cols = ['absences', 'failures', 'G1', 'G2']
        if all(col in X_out.columns for col in risk_cols):
            X_out['academic_risk_score'] = (X_out['absences'] * 0.4) + (X_out['failures'] * 0.4) - ((X_out['G1'] + X_out['G2']) * 0.2)
            
        # 4. family_background_score
        if all(col in X_out.columns for col in ['Medu', 'Fedu']):
            X_out['family_background_score'] = (X_out['Medu'] + X_out['Fedu']) / 2.0
            
        # 5. engagement_index
        eng_cols = ['activities', 'nursery', 'higher']
        if all(col in X_out.columns for col in eng_cols):
            X_out['engagement_index'] = sum((X_out[col] == 'yes').astype(int) for col in eng_cols)
            
        # 6. interaction_absences_failures
        if all(col in X_out.columns for col in ['absences', 'failures']):
            X_out['interaction_absences_failures'] = X_out['absences'] * X_out['failures']
            
        # 7. interaction_study_higher
        if all(col in X_out.columns for col in ['studytime', 'higher']):
            X_out['interaction_study_higher'] = X_out['studytime'] * (X_out['higher'] == 'yes').astype(int)
            
        # 8. interaction_support_internet
        if 'support_index' in X_out.columns and 'internet' in X_out.columns:
            X_out['interaction_support_internet'] = X_out['support_index'] * (X_out['internet'] == 'yes').astype(int)
            
        # 9. health_social_index
        if all(col in X_out.columns for col in ['health', 'goout']):
            X_out['health_social_index'] = X_out['health'] * X_out['goout']
            
        # 10. alcohol_consumption_index
        if all(col in X_out.columns for col in ['Dalc', 'Walc']):
            X_out['alcohol_consumption_index'] = X_out['Dalc'] + X_out['Walc']
            
        return X_out
