"""
Utilities for model loading, prediction, and evaluation for the Stroke Risk Prediction application.
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
import os
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, 
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import logging

from src.config import MODEL_PATH, RISK_CATEGORIES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model():
    """
    Load the XGBoost model from the specified path.
    
    Returns:
        object: The loaded XGBoost model
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
        Exception: For other errors during model loading
    """
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        logger.info(f"Loading model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully")
        
        # Log feature names if available for debugging
        if hasattr(model, 'feature_names_in_'):
            logger.info(f"Model feature names: {model.feature_names_in_}")
        elif hasattr(model, 'get_booster'):
            logger.info(f"XGBoost feature names: {model.get_booster().feature_names}")
            
        return model
    
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {str(e)}")
        raise
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise Exception(f"Failed to load model: {str(e)}")

def predict_stroke_risk(model, input_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make stroke risk predictions using the loaded model.
    
    Args:
        model: The loaded XGBoost model
        input_data (pd.DataFrame): Preprocessed input data
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing binary predictions and probability scores
        
    Raises:
        Exception: If prediction fails
    """
    try:
        # Make probability predictions
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_data)[:, 1]
            # Convert to binary predictions (threshold 0.5)
            predictions = (probabilities > 0.5).astype(int)
        else:
            # Fallback if model doesn't have predict_proba
            predictions = model.predict(input_data)
            probabilities = predictions.astype(float)
            
        return predictions, probabilities
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise Exception(f"Failed to make prediction: {str(e)}")

def get_risk_category(probability: float) -> Tuple[str, str, str]:
    """
    Get risk category based on probability.
    
    Returns:
        Tuple[str, str, str]: (category_name, color, description)
    """
    if probability < 0.1:
        return "Low Risk", "green", "Low risk of stroke."
    elif probability < 0.3:
        return "Moderate Risk", "orange", "Moderate risk of stroke."
    else:
        return "High Risk", "red", "High risk of stroke."

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
    """
    Evaluate the model performance.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_prob (np.ndarray, optional): Predicted probabilities
        
    Returns:
        Dict[str, Any]: Dictionary of evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    if y_prob is not None:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
        
        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        metrics['precision_curve'] = precision
        metrics['recall_curve'] = recall
    
    return metrics

def get_feature_importance(model) -> pd.DataFrame:
    """
    Extract feature importance from the model.
    
    Args:
        model: The loaded XGBoost model
        
    Returns:
        pd.DataFrame: DataFrame with feature names and importance scores
    """
    try:
        if hasattr(model, 'feature_importances_'):
            # For scikit-learn compatible models
            importance = model.feature_importances_
            
            if hasattr(model, 'feature_names_in_'):
                features = model.feature_names_in_
            else:
                # Fallback if feature names not available
                features = [f"feature_{i}" for i in range(len(importance))]
                
        elif hasattr(model, 'get_booster'):
            # For XGBoost models
            importance_dict = model.get_booster().get_score(importance_type='weight')
            features = list(importance_dict.keys())
            importance = [importance_dict[feature] for feature in features]
        else:
            logger.warning("Could not extract feature importance from model")
            return pd.DataFrame({"Feature": ["Unknown"], "Importance": [0]})
        
        # Create and return DataFrame
        return pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values("Importance", ascending=False)
        
    except Exception as e:
        logger.error(f"Error extracting feature importance: {str(e)}")
        return pd.DataFrame({"Feature": ["Error"], "Importance": [0]})