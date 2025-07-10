"""
Main application file for the Stroke Risk Prediction Streamlit app.
"""

import streamlit as st
import os
import sys
import logging
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

# Import utilities
from src.utils import setup_page_config, load_css, check_models_available

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to run the Stroke Risk Prediction application."""
    # Set up page configuration
    setup_page_config()
    
    # Load custom CSS
    load_css()
    
    # Check if model is available
    if not check_models_available():
        st.error("⚠️ Error: The XGBoost model file could not be found.")
        st.info("Expected location: ./models/xgboost_stroke_model.joblib")
        st.info("Please ensure the model file exists and try again.")
        return
    
    # Main page content
    st.title("🧠 Stroke Risk Prediction System")
    
    st.markdown("""
    Welcome to the Stroke Risk Prediction System! This application uses machine learning 
    to assess stroke risk based on patient health data.
    
    ### 🔍 Available Features:
    
    - **Individual Prediction**: Assess stroke risk for a single patient
    - **Batch Prediction**: Process multiple patient records (for medical professionals)  
    - **Model Performance**: View detailed model performance metrics
    - **About**: Learn more about the application and methodology
    
    ### 📋 How to Use:
    
    1. **Navigate** to the desired page using the sidebar
    2. **Enter patient data** or upload a CSV file
    3. **Get predictions** with detailed risk assessments
    4. **Download reports** for record keeping
    
    ### ⚠️ Important Disclaimer:
    
    This tool is for **educational and research purposes only**. It should not replace 
    professional medical advice, diagnosis, or treatment. Always consult with qualified 
    healthcare professionals for medical decisions.
    """)
    
    # Quick stats or model info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", "XGBoost")
    
    with col2:
        st.metric("Model Accuracy", "92%")
    
    with col3:
        st.metric("Features Used", "10")
    
    # Instructions for navigation
    st.info("""
    💡 **Tip**: Use the sidebar navigation to explore different features of the application.
    Start with 'Individual Prediction' to assess stroke risk for a single patient.
    """)

if __name__ == "__main__":
    main()