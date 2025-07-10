# // filepath: src/visualization.py
"""
Visualization utilities for the Stroke Risk Prediction application.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any, Optional
import streamlit as st
import base64
from io import BytesIO

def create_risk_gauge(probability: float) -> go.Figure:
    """
    Create a gauge chart visualizing stroke risk probability.
    
    Args:
        probability (float): Risk probability from 0 to 1
        
    Returns:
        go.Figure: Plotly gauge chart figure
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Stroke Risk Level", 'font': {'size': 24}},
        delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#92d14f'},     # Green
                {'range': [20, 40], 'color': '#c7e596'},    # Light green
                {'range': [40, 60], 'color': '#fee08b'},    # Yellow
                {'range': [60, 80], 'color': '#fdae61'},    # Orange
                {'range': [80, 100], 'color': '#f46d43'}    # Red
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    return fig

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Create a horizontal bar chart of feature importances.
    
    Args:
        importance_df (pd.DataFrame): DataFrame with Feature and Importance columns
        top_n (int): Number of top features to display
        
    Returns:
        go.Figure: Plotly bar chart of feature importances
    """
    # Take top n features
    plot_df = importance_df.head(top_n)
    
    fig = px.bar(
        plot_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        labels={'Importance': 'Relative Importance', 'Feature': 'Feature'},
        title='Top Factors Influencing Stroke Risk',
        color='Importance',
        color_continuous_scale=px.colors.sequential.Blues,
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig

def plot_confusion_matrix(cm: np.ndarray) -> plt.Figure:
    """
    Create a confusion matrix visualization.
    
    Args:
        cm (np.ndarray): Confusion matrix array from sklearn
        
    Returns:
        plt.Figure: Matplotlib figure with confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        cbar=False,
        xticklabels=["No Stroke", "Stroke"],
        yticklabels=["No Stroke", "Stroke"]
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    
    return plt.gcf()

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> go.Figure:
    """
    Create an ROC curve visualization.
    
    Args:
        fpr (np.ndarray): False positive rates
        tpr (np.ndarray): True positive rates
        auc (float): Area under the curve value
        
    Returns:
        go.Figure: Plotly figure with ROC curve
    """
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, 
        y=tpr,
        mode='lines',
        name=f'XGBoost (AUC = {auc:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    # Add diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        margin=dict(l=40, r=40, t=40, b=40),
        height=450,
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig

def plot_precision_recall_curve(precision: np.ndarray, recall: np.ndarray) -> go.Figure:
    """
    Create a Precision-Recall curve visualization.
    
    Args:
        precision (np.ndarray): Precision values
        recall (np.ndarray): Recall values
        
    Returns:
        go.Figure: Plotly figure with Precision-Recall curve
    """
    fig = go.Figure()
    
    # Add PR curve
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name='XGBoost',
        line=dict(color='green', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        yaxis=dict(range=[0, 1.05]),
        margin=dict(l=40, r=40, t=40, b=40),
        height=450,
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig

def plot_prediction_distribution(probabilities: np.ndarray) -> go.Figure:
    """
    Create a histogram of prediction probabilities.
    
    Args:
        probabilities (np.ndarray): Array of prediction probabilities
        
    Returns:
        go.Figure: Plotly histogram
    """
    fig = px.histogram(
        x=probabilities,
        nbins=20,
        labels={'x': 'Stroke Risk Probability'},
        title='Distribution of Stroke Risk Predictions',
        color_discrete_sequence=['blue']
    )
    
    fig.update_layout(
        xaxis_title='Stroke Risk Probability',
        yaxis_title='Count',
        xaxis=dict(range=[0, 1]),
        bargap=0.2,
        height=400,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    # Add a vertical line at the threshold (0.5)
    fig.add_vline(x=0.5, line_width=2, line_dash="dash", line_color="red")
    fig.add_annotation(x=0.5, y=0, text="Threshold (0.5)", showarrow=False, yshift=10)
    
    return fig

def get_binary_file_downloader_html(fig, filename, text):
    """
    Generate HTML code for downloading a matplotlib figure.
    
    Args:
        fig (plt.Figure): Matplotlib figure to download
        filename (str): Name for the downloaded file
        text (str): Link text to display
    
    Returns:
        str: HTML download link
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    data = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{data}" download="{filename}">{text}</a>'
    return href

def fig_to_base64(fig):
    """
    Convert a matplotlib figure to base64 encoded string.
    
    Args:
        fig (plt.Figure): Matplotlib figure to encode
        
    Returns:
        str: Base64 encoded string
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    return base64.b64encode(buf.getvalue()).decode()