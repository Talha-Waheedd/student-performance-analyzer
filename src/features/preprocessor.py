import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE
import logging

logger = logging.getLogger(__name__)

class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps outliers using IQR or Z-score method.
    """
    def __init__(self, method='iqr', factor=1.5, columns=None):
        self.method = method
        self.factor = factor
        self.columns = columns
        self.bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        cols_to_process = self.columns if self.columns else X_df.select_dtypes(include=[np.number]).columns

        for col in cols_to_process:
            if col in X_df.columns:
                if self.method == 'iqr':
                    Q1 = X_df[col].quantile(0.25)
                    Q3 = X_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.factor * IQR
                    upper_bound = Q3 + self.factor * IQR
                elif self.method == 'zscore':
                    mean = X_df[col].mean()
                    std = X_df[col].std()
                    lower_bound = mean - self.factor * std
                    upper_bound = mean + self.factor * std
                else:
                    raise ValueError("Method must be 'iqr' or 'zscore'")
                
                self.bounds_[col] = (lower_bound, upper_bound)
        return self

    def transform(self, X):
        X_out = pd.DataFrame(X).copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in X_out.columns:
                X_out[col] = np.clip(X_out[col], lower, upper)
        return X_out

def build_preprocessor(numeric_features, categorical_features, config):
    """
    Builds a scikit-learn ColumnTransformer for preprocessing.
    """
    impute_strategy = config.get('imputation_strategy', 'median')
    scale_method = config.get('scaling_method', 'standard')

    # Numeric Pipeline Steps
    if impute_strategy == 'knn':
        numeric_imputer = KNNImputer(n_neighbors=5)
    else:
        numeric_imputer = SimpleImputer(strategy=impute_strategy)

    if scale_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    from sklearn.pipeline import Pipeline
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', numeric_imputer),
        ('scaler', scaler)
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def handle_class_imbalance(X, y, method='smote'):
    if method == 'smote':
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res
    return X, y
