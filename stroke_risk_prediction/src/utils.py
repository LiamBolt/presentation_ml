"""
Utility functions for the Stroke Risk Prediction application.
"""

import streamlit as st
import os
from pathlib import Path
import joblib
import logging

# Import from existing modules
from .model_utils import load_model, get_risk_category
from .data_processor import preprocess_input
from .config import XGBOOST_MODEL_PATH, CHATBOT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

def setup_page_config():
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title="Stroke Risk Prediction",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def load_css():
    """Load custom CSS styling for the application."""
    # First try to load external CSS file
    css_file = Path("assets/styles.css")
    if css_file.exists():
        with open(css_file) as f:
            external_css = f.read()
        st.markdown(f"<style>{external_css}</style>", unsafe_allow_html=True)
    
    # Then add inline CSS for critical styles
    css = """
    <style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    
    .risk-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    .low-risk {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .moderate-risk {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .high-risk {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background-color: #2E86AB;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #1a5f7a;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def check_models_available():
    """Check if the required model file is available."""
    model_path = Path(XGBOOST_MODEL_PATH)
    
    if not model_path.exists():
        logger.error(f"Model file not found at: {model_path}")
        return False
    
    try:
        # Try to load the model to ensure it's valid
        model = joblib.load(model_path)
        logger.info("Model successfully loaded and verified")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def format_patient_data(data):
    """Format patient data for display."""
    formatted = {}
    
    # Format each field for better display
    formatted["Age"] = f"{data['age']} years"
    formatted["Gender"] = data['gender']
    formatted["Hypertension"] = "Yes" if data['hypertension'] == 1 else "No"
    formatted["Heart Disease"] = "Yes" if data['heart_disease'] == 1 else "No"
    formatted["Ever Married"] = data['ever_married']
    formatted["Work Type"] = data['work_type']
    formatted["Residence Type"] = data['Residence_type']
    formatted["Average Glucose Level"] = f"{data['avg_glucose_level']:.1f} mg/dL"
    formatted["BMI"] = f"{data['bmi']:.1f}"
    formatted["Smoking Status"] = data['smoking_status']
    
    return formatted

def validate_input_data(data):
    """Validate input data."""
    try:
        # Check age
        if data['age'] < 0 or data['age'] > 120:
            return False, "Age must be between 0 and 120"
        
        # Check glucose level
        if data['avg_glucose_level'] < 0 or data['avg_glucose_level'] > 500:
            return False, "Glucose level must be between 0 and 500"
        
        # Check BMI
        if data['bmi'] < 10 or data['bmi'] > 100:
            return False, "BMI must be between 10 and 100"
        
        return True, ""
    except Exception as e:
        return False, str(e)

def create_download_link(data, filename, text="Download"):
    """Create a download link for data."""
    import base64
    import io
    
    if isinstance(data, str):
        # For text data
        b64 = base64.b64encode(data.encode()).decode()
        href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">{text}</a>'
    else:
        # For binary data (like PDF)
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'
    
    return href

def create_report_pdf(predictions_df, summary_stats):
    """Create a PDF report for batch predictions."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    import io
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story.append(Paragraph("Stroke Risk Prediction Report", title_style))
    story.append(Spacer(1, 20))
    
    # Summary
    story.append(Paragraph("Summary Statistics", styles['Heading2']))
    summary_data = [
        ['Total Predictions', str(summary_stats['total'])],
        ['Low Risk', str(summary_stats['low_risk'])],
        ['Moderate Risk', str(summary_stats['moderate_risk'])],
        ['High Risk', str(summary_stats['high_risk'])],
        ['Average Risk Score', f"{summary_stats['avg_risk']:.3f}"]
    ]
    
    summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Detailed predictions table
    story.append(Paragraph("Detailed Predictions", styles['Heading2']))
    
    # Prepare data for table
    table_data = [['ID', 'Age', 'Gender', 'Risk Score', 'Risk Category']]
    for idx, row in predictions_df.iterrows():
        table_data.append([
            str(idx + 1),
            str(row.get('age', 'N/A')),
            str(row.get('gender', 'N/A')),
            f"{row['risk_score']:.3f}",
            row['risk_category']
        ])
    
    predictions_table = Table(table_data, colWidths=[0.5*inch, 0.8*inch, 1*inch, 1*inch, 1.2*inch])
    predictions_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
    ]))
    
    story.append(predictions_table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
