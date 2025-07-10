"""
PDF report generation utilities for the Stroke Risk Prediction application.
"""

import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
import streamlit as st
from fpdf import FPDF
import tempfile
from typing import Dict, Any, List, Optional, Union

from src.config import RISK_CATEGORIES, RECOMMENDATIONS, RISK_FACTOR_DESCRIPTIONS
from src.visualization import fig_to_base64

class PDF(FPDF):
    """Extended FPDF class with header and footer"""
    def header(self):
        # Logo (optional)
        # self.image('logo.png', 10, 8, 33)
        
        # Set font for header
        self.set_font('Arial', 'B', 15)
        
        # Title
        self.cell(0, 10, 'Stroke Risk Prediction Report', 0, 1, 'C')
        
        # Line break
        self.ln(10)
    
    def footer(self):
        # Set position at 1.5 cm from bottom
        self.set_y(-15)
        
        # Set font for footer
        self.set_font('Arial', 'I', 8)
        
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
        
        # Disclaimer
        self.set_y(-10)
        self.cell(0, 10, 'Disclaimer: This report is for informational purposes only and does not constitute medical advice.', 0, 0, 'C')

def generate_individual_report(
    input_data: Dict[str, Any], 
    prediction: int, 
    probability: float,
    feature_importance: pd.DataFrame
) -> BytesIO:
    """
    Generate a PDF report for an individual prediction.
    
    Args:
        input_data (Dict[str, Any]): User input data
        prediction (int): Binary prediction (0 or 1)
        probability (float): Prediction probability
        feature_importance (pd.DataFrame): Feature importance data
        
    Returns:
        BytesIO: PDF file as BytesIO object
    """
    # Create PDF object
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Set font
    pdf.set_font('Arial', '', 12)
    
    # Add date and time
    now = datetime.now()
    pdf.cell(0, 10, f'Generated on: {now.strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
    pdf.ln(5)
    
    # Add Risk Assessment
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Stroke Risk Assessment', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    # Risk level
    risk_category = next((cat for cat, (low, high) in RISK_CATEGORIES.items() 
                          if low <= probability < high), "Unknown")
    
    # Different formatting based on risk level
    if prediction == 1:
        pdf.set_text_color(255, 0, 0)  # Red
        pdf.cell(0, 10, f'Risk Level: {risk_category} ({probability:.1%})', 0, 1)
    else:
        pdf.set_text_color(0, 128, 0)  # Green
        pdf.cell(0, 10, f'Risk Level: {risk_category} ({probability:.1%})', 0, 1)
    
    # Reset text color
    pdf.set_text_color(0, 0, 0)
    
    # Add recommendations
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Recommendations', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    if risk_category in RECOMMENDATIONS:
        for i, recommendation in enumerate(RECOMMENDATIONS[risk_category]):
            pdf.cell(0, 8, f"{i+1}. {recommendation}", 0, 1)
    
    # Add patient information
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Patient Information', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    # Format the user input data
    formatted_input = {
        'Age': input_data.get('age', 'N/A'),
        'Gender': input_data.get('gender', 'N/A'),
        'Hypertension': 'Yes' if input_data.get('hypertension', 0) == 1 else 'No',
        'Heart Disease': 'Yes' if input_data.get('heart_disease', 0) == 1 else 'No',
        'Ever Married': input_data.get('ever_married', 'N/A'),
        'Work Type': input_data.get('work_type', 'N/A'),
        'Residence Type': input_data.get('Residence_type', 'N/A'),
        'Avg Glucose Level': f"{input_data.get('avg_glucose_level', 'N/A')} mg/dL",
        'BMI': input_data.get('bmi', 'N/A'),
        'Smoking Status': input_data.get('smoking_status', 'N/A')
    }
    
    # Display patient information
    for key, value in formatted_input.items():
        pdf.cell(0, 8, f"{key}: {value}", 0, 1)
    
    # Add key risk factors
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Key Risk Factors', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    # Get top 5 risk factors
    top_factors = feature_importance.head(5)
    for _, row in top_factors.iterrows():
        factor = row['Feature']
        importance = row['Importance']
        
        # Get description for this risk factor
        description = RISK_FACTOR_DESCRIPTIONS.get(factor, "")
        
        pdf.cell(0, 8, f"• {factor.capitalize()}: {description}", 0, 1)
    
    # Add important disclaimer
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Important Notice', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 6, 'This risk assessment is based on a machine learning model and should not replace professional medical advice. The model provides an estimate based on limited data points and may not capture all relevant factors. Always consult with a qualified healthcare provider for proper evaluation, diagnosis, and treatment recommendations.')
    
    # Get PDF as BytesIO object
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    
    return pdf_output

def generate_batch_report(
    results_df: pd.DataFrame,
    metrics: Dict[str, Any],
    figures: Dict[str, Union[plt.Figure, BytesIO]]
) -> BytesIO:
    """
    Generate a PDF report for batch predictions.
    
    Args:
        results_df (pd.DataFrame): DataFrame with prediction results
        metrics (Dict[str, Any]): Model performance metrics
        figures (Dict[str, Union[plt.Figure, BytesIO]]): Dictionary of figures to include
        
    Returns:
        BytesIO: PDF file as BytesIO object
    """
    # Create PDF object
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Set font
    pdf.set_font('Arial', '', 12)
    
    # Add date and time
    now = datetime.now()
    pdf.cell(0, 10, f'Generated on: {now.strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
    pdf.ln(5)
    
    # Add summary statistics
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Summary Statistics', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    # Number of patients
    total_patients = len(results_df)
    high_risk = (results_df['Prediction'] == 1).sum()
    high_risk_pct = high_risk / total_patients * 100 if total_patients > 0 else 0
    
    pdf.cell(0, 8, f"Total patients: {total_patients}", 0, 1)
    pdf.cell(0, 8, f"High risk patients: {high_risk} ({high_risk_pct:.1f}%)", 0, 1)
    pdf.cell(0, 8, f"Average risk probability: {results_df['Probability'].mean():.1%}", 0, 1)
    
    # Add model performance metrics
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Model Performance', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    # Only add metrics if they exist in the dictionary
    if 'accuracy' in metrics:
        pdf.cell(0, 8, f"Accuracy: {metrics['accuracy']:.3f}", 0, 1)
    if 'precision' in metrics:
        pdf.cell(0, 8, f"Precision: {metrics['precision']:.3f}", 0, 1)
    if 'recall' in metrics:
        pdf.cell(0, 8, f"Recall: {metrics['recall']:.3f}", 0, 1)
    if 'f1' in metrics:
        pdf.cell(0, 8, f"F1 Score: {metrics['f1']:.3f}", 0, 1)
    if 'auc' in metrics:
        pdf.cell(0, 8, f"AUC: {metrics['auc']:.3f}", 0, 1)
    
    # Add explanation
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Understanding the Results', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    explanation = """
    The prediction results include:
    
    • Prediction: Binary outcome (1 = High stroke risk, 0 = Low stroke risk)
    • Probability: The model's confidence in the prediction (0-100%)
    • Risk Category: Classification based on probability thresholds
    
    Patients classified as "High Risk" or above should be considered for further evaluation.
    """
    
    pdf.multi_cell(0, 6, explanation)
    
    # Add visualizations if available
    if figures:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Visualizations', 0, 1)
        
        # Add confusion matrix if available
        if 'confusion_matrix' in figures:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Confusion Matrix', 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 5, 'Shows the count of correct and incorrect predictions. The diagonal represents correct predictions.')
            
            # Save confusion matrix to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
                if isinstance(figures['confusion_matrix'], plt.Figure):
                    figures['confusion_matrix'].savefig(temp_file.name, format='png', dpi=300, bbox_inches='tight')
                    pdf.image(temp_file.name, x=40, y=pdf.get_y(), w=120)
                elif isinstance(figures['confusion_matrix'], BytesIO):
                    figures['confusion_matrix'].seek(0)
                    pdf.image(figures['confusion_matrix'], x=40, y=pdf.get_y(), w=120)
                    
            pdf.ln(100)  # Space for the image
        
        # Add ROC curve if available
        if 'roc_curve' in figures:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'ROC Curve', 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 5, 'Shows the trade-off between sensitivity (TPR) and specificity (1-FPR). A curve closer to the top-left corner indicates better performance.')
            
            # Save ROC curve to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
                if isinstance(figures['roc_curve'], plt.Figure):
                    figures['roc_curve'].savefig(temp_file.name, format='png', dpi=300, bbox_inches='tight')
                    pdf.image(temp_file.name, x=40, y=pdf.get_y(), w=120)
                elif isinstance(figures['roc_curve'], BytesIO):
                    figures['roc_curve'].seek(0)
                    pdf.image(figures['roc_curve'], x=40, y=pdf.get_y(), w=120)
                    
            pdf.ln(100)  # Space for the image
    
    # Add recommendations
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Recommendations for Follow-up', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    recommendations = """
    Based on the prediction results, consider the following actions:
    
    1. High and Very High Risk Patients:
       • Schedule prompt clinical evaluation
       • Assess for modifiable risk factors
       • Consider advanced diagnostic testing
    
    2. Elevated Risk Patients:
       • Clinical evaluation within 1-2 months
       • Focus on lifestyle modifications
       • Monitor blood pressure and glucose levels regularly
    
    3. Low and Moderate Risk Patients:
       • Routine follow-up
       • Patient education on stroke warning signs
       • Annual risk assessment
    """
    
    pdf.multi_cell(0, 6, recommendations)
    
    # Add important disclaimer
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Important Notice', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 6, 'This risk assessment is based on a machine learning model and should be used as a screening tool only. Clinical judgment should always take precedence over model predictions. The model has limitations and may not capture all relevant factors for stroke risk assessment.')
    
    # Get PDF as BytesIO object
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    
    return pdf_output

def get_pdf_download_link(pdf_bytes: BytesIO, filename: str) -> str:
    """
    Generate HTML link for downloading a PDF.
    
    Args:
        pdf_bytes (BytesIO): PDF file as BytesIO object
        filename (str): Name for the downloaded file
        
    Returns:
        str: HTML for download link
    """
    b64 = base64.b64encode(pdf_bytes.read()).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF Report</a>'