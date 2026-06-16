import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import plotly.express as px
import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred, y_prob=None):
    """
    Compute comprehensive metrics for classification.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            logger.warning("ROC AUC could not be calculated.")
            metrics['roc_auc'] = None
    
    return metrics

def plot_confusion_matrix_plotly(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    z_text = [[str(y) for y in x] for x in cm]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted 0', 'Predicted 1'],
        y=['Actual 0', 'Actual 1'],
        hoverongaps=False,
        text=z_text,
        texttemplate="%{text}",
        colorscale='Blues'
    ))
    fig.update_layout(title=title)
    return fig

def plot_roc_curve_plotly(y_true, y_prob, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    fig = px.area(
        x=fpr, y=tpr,
        title=f"{title} (AUC={auc_score:.4f})",
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    return fig
