import os
import joblib
import logging
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import mlflow
import mlflow.sklearn

from src.data.data_loader import load_data, validate_data
from src.features.feature_engineering import FeatureEngineer
from src.features.preprocessor import build_preprocessor, handle_class_imbalance, OutlierCapper
from src.evaluation.metrics import evaluate_model

logger = logging.getLogger(__name__)

class StudentPerformancePipeline:
    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self.models = {}
        self.best_model_name = None
        
    def prepare_data(self, df):
        """
        Extract features and target.
        """
        target_col = self.config['pipeline'].get('target_col', 'passed')
        if target_col not in df.columns:
            # Maybe it needs to be created from G3
            if 'G3' in df.columns:
                df[target_col] = (df['G3'] >= 10).astype(int)
        
        X = df.drop(columns=[target_col, 'G3'], errors='ignore')
        y = df[target_col]
        return X, y

    def get_base_models(self):
        """
        Return the 6 base models with their hyperparameter grids.
        """
        return {
            'LogisticRegression': {
                'model': LogisticRegression(max_iter=1000, random_state=self.config['pipeline'].get('random_state', 42)),
                'params': {'classifier__C': [0.01, 0.1, 1, 10, 100], 'classifier__penalty': ['l2']}
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=self.config['pipeline'].get('random_state', 42)),
                'params': {'classifier__n_estimators': [100], 'classifier__max_depth': [None, 10, 20], 'classifier__min_samples_split': [2, 5, 10]}
            },
            'XGBoost': {
                'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=self.config['pipeline'].get('random_state', 42)),
                'params': {'classifier__learning_rate': [0.01, 0.1], 'classifier__max_depth': [3, 5, 7]}
            },
            'LightGBM': {
                'model': LGBMClassifier(random_state=self.config['pipeline'].get('random_state', 42)),
                'params': {'classifier__num_leaves': [31, 50], 'classifier__learning_rate': [0.01, 0.1]}
            },
            'SVM': {
                'model': SVC(probability=True, random_state=self.config['pipeline'].get('random_state', 42)),
                'params': {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']}
            },
            'NeuralNetwork': {
                'model': MLPClassifier(max_iter=1000, random_state=self.config['pipeline'].get('random_state', 42)),
                'params': {'classifier__hidden_layer_sizes': [(100, 50)], 'classifier__alpha': [0.0001, 0.001], 'classifier__learning_rate_init': [0.001, 0.01]}
            }
        }

    def train(self, X_train, y_train):
        logger.info("Building preprocessing pipeline...")
        
        # Identify numeric and categorical columns dynamically
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        preprocessor = build_preprocessor(numeric_features, categorical_features, self.config['pipeline'])
        
        # We need to construct the full sklearn Pipeline
        # Including FeatureEngineer, OutlierCapper, Preprocessor, and SMOTE
        base_steps = [
            ('feature_engineer', FeatureEngineer()),
            ('outlier_capper', OutlierCapper(method=self.config['pipeline'].get('outlier_method', 'iqr'))),
            ('preprocessor', preprocessor)
        ]
        
        if self.config['pipeline'].get('handle_imbalance', 'smote') == 'smote':
            from imblearn.over_sampling import SMOTE
            base_steps.append(('smote', SMOTE(random_state=self.config['pipeline'].get('random_state', 42))))
        
        models_dict = self.get_base_models()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config['pipeline'].get('random_state', 42))
        
        best_score = 0
        
        for model_name, model_info in models_dict.items():
            logger.info(f"Training {model_name}...")
            full_pipeline = Pipeline(steps=base_steps + [('classifier', model_info['model'])])
            
            with mlflow.start_run(run_name=model_name):
                mlflow.autolog(disable=True) # Manual logging for better control with GridSearchCV
                
                grid_search = GridSearchCV(
                    full_pipeline, 
                    model_info['params'], 
                    cv=cv, 
                    scoring='roc_auc', 
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                
                # Log best params and metrics
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metric("best_cv_roc_auc", grid_search.best_score_)
                mlflow.sklearn.log_model(grid_search.best_estimator_, artifact_path="model")
                
                self.models[model_name] = grid_search.best_estimator_
                
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    self.best_model_name = model_name

        logger.info(f"Best base model: {self.best_model_name} with ROC-AUC: {best_score:.4f}")
        
        # Ensemble: Voting & Stacking (using top 3 models - assuming RF, XGB, LGBM for simplicity here)
        top_models = [
            ('rf', self.models['RandomForest'].named_steps['classifier']),
            ('xgb', self.models['XGBoost'].named_steps['classifier']),
            ('lgbm', self.models['LightGBM'].named_steps['classifier'])
        ]
        
        logger.info("Training Voting Classifier...")
        voting_clf = VotingClassifier(estimators=top_models, voting='soft')
        voting_pipeline = Pipeline(steps=base_steps + [('classifier', voting_clf)])
        voting_pipeline.fit(X_train, y_train)
        self.models['VotingEnsemble'] = voting_pipeline
        
        logger.info("Training Stacking Classifier...")
        stacking_clf = StackingClassifier(estimators=top_models, final_estimator=LogisticRegression())
        stacking_pipeline = Pipeline(steps=base_steps + [('classifier', stacking_clf)])
        stacking_pipeline.fit(X_train, y_train)
        self.models['StackingEnsemble'] = stacking_pipeline
        
        # Set the main pipeline to the best base model for now
        self.pipeline = self.models[self.best_model_name]
        return self

    def predict(self, X):
        if self.pipeline is None:
            raise ValueError("Pipeline is not trained yet.")
        return self.pipeline.predict(X)
        
    def predict_proba(self, X):
        if self.pipeline is None:
            raise ValueError("Pipeline is not trained yet.")
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'models': self.models,
            'best_model_name': self.best_model_name
        }, filepath)
        logger.info(f"Models saved to {filepath}")

    def load(self, filepath):
        data = joblib.load(filepath)
        self.models = data['models']
        self.best_model_name = data['best_model_name']
        self.pipeline = self.models[self.best_model_name]
        logger.info(f"Models loaded from {filepath}")
