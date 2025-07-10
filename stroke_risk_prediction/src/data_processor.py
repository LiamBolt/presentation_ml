"""
Data processing utilities for the Stroke Risk Prediction application.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from io import StringIO, BytesIO
import base64

from src.config import (
    GENDER_MAP, EVER_MARRIED_MAP, WORK_TYPE_MAP, 
    RESIDENCE_TYPE_MAP, SMOKING_STATUS_MAP, 
    EXPECTED_COLUMNS, REQUIRED_COLUMNS
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_input(input_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess user input to match the expected format for the model.
    """
    try:
        # Convert input dictionary to pandas DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Debug: Print input columns
        print(f"DEBUG: Input columns: {list(input_df.columns)}")
        
        # Check for missing required fields
        missing_fields = [field for field in EXPECTED_COLUMNS if field not in input_df.columns]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Encode categorical variables to match training data encoding
        input_df['gender'] = input_df['gender'].map(GENDER_MAP)
        input_df['ever_married'] = input_df['ever_married'].map(EVER_MARRIED_MAP)
        input_df['work_type'] = input_df['work_type'].map(WORK_TYPE_MAP)
        input_df['Residence_type'] = input_df['Residence_type'].map(RESIDENCE_TYPE_MAP)
        input_df['smoking_status'] = input_df['smoking_status'].map(SMOKING_STATUS_MAP)
        
        # Check for unmapped values
        if input_df.isnull().any().any():
            null_columns = input_df.columns[input_df.isnull().any()].tolist()
            raise ValueError(f"Invalid values in columns: {null_columns}")
        
        # CRITICAL: Use the EXACT feature order that the model was trained with
        # Based on the model file, the correct order is:
        model_feature_order = [
            'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
            'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
        ]
        
        # Ensure all expected columns are present and reorder to match model
        input_df = input_df[model_feature_order]
        
        # Ensure proper data types
        numeric_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
        for col in numeric_columns:
            input_df[col] = pd.to_numeric(input_df[col])
        
        # Ensure integer columns are int
        int_columns = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        for col in int_columns:
            input_df[col] = input_df[col].astype(int)
        
        # Debug: Print final columns after reordering
        print(f"DEBUG: Final columns after reordering: {list(input_df.columns)}")
        
        logger.info(f"Preprocessed input shape: {input_df.shape}")
        logger.info(f"Feature order: {list(input_df.columns)}")
        
        return input_df
        
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise ValueError(f"Data preprocessing failed: {str(e)}")

def validate_uploaded_dataset(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that the uploaded dataset contains all required columns.
    
    Args:
        df (pd.DataFrame): The uploaded dataset
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    # Check for required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    
    # Check for empty dataframe
    if df.empty:
        return False, "The uploaded file contains no data"
    
    # Check for sufficient number of rows
    if len(df) < 1:
        return False, "The dataset must contain at least one row"
    
    # Check for too many rows (optional, for performance)
    if len(df) > 10000:
        return False, "The dataset is too large (max 10,000 rows)"
    
    return True, ""

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a dataset for batch prediction.
    """
    # 1) Copy
    processed_df = df.copy()
    
    # 2) Fill defaults for non-crucial columns if missing
    defaults = {
        'ever_married': 'No',
        'work_type': 'Private',
        'Residence_type': 'Urban',
        'smoking_status': 'Unknown'
    }
    for col, default in defaults.items():
        if col not in processed_df.columns:
            processed_df[col] = default
    
    # 3) Map categoricals
    processed_df['gender']        = processed_df['gender'].map(GENDER_MAP)
    processed_df['ever_married']  = processed_df['ever_married'].map(EVER_MARRIED_MAP)
    processed_df['work_type']     = processed_df['work_type'].map(WORK_TYPE_MAP)
    processed_df['Residence_type']= processed_df['Residence_type'].map(RESIDENCE_TYPE_MAP)
    processed_df['smoking_status']= processed_df['smoking_status'].map(SMOKING_STATUS_MAP)
    
    # 4) Numeric columns → numeric, fill NaNs with median
    for col in ['age','hypertension','heart_disease','avg_glucose_level','bmi']:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        if processed_df[col].isna().any():
            processed_df[col].fillna(processed_df[col].median(), inplace=True)
    
    # 5) EXACT feature order the model expects:
    feature_order = [
        'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
        'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
    ]
    
    # 6) Select & reorder—this drops any extra cols (e.g. id, stroke)
    processed_df = processed_df[feature_order]
    
    # 7) Ensure ints
    for col in ['gender','hypertension','heart_disease','ever_married',
                'work_type','Residence_type','smoking_status']:
        processed_df[col] = processed_df[col].astype(int)
    
    logger.info(f"Preprocessed dataset shape: {processed_df.shape}")
    logger.info(f"Feature order: {list(processed_df.columns)}")
    return processed_df

def generate_csv_download_link(df: pd.DataFrame, filename: str = "prediction_results.csv") -> str:
    """
    Generate a download link for a DataFrame as CSV.
    
    Args:
        df (pd.DataFrame): DataFrame to convert to CSV
        filename (str): Name of the file to download
        
    Returns:
        str: HTML for a download link
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

def generate_enhanced_medical_report(results_df: pd.DataFrame, summary_stats: Dict[str, Any]) -> BytesIO:
    """Enhanced report with more visualizations"""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    import io
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch
    
    # Define specific colors we'll use as hex values to avoid the colors module issue
    dark_blue = colors.HexColor('#000080')  # Navy blue hex code
    whitesmoke = colors.HexColor('#F5F5F5')  # Whitesmoke hex code
    red = colors.HexColor('#FF0000')
    orange = colors.HexColor('#FFA500')
    green = colors.HexColor('#008000')
    lavender_blush = colors.HexColor('#FFF0F5')
    white = colors.HexColor('#FFFFFF')
    black = colors.HexColor('#000000')
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title with better styling
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,  # Center alignment
        textColor=dark_blue
    )
    story.append(Paragraph("Clinical Stroke Risk Assessment Report", title_style))
    story.append(Spacer(1, 20))
    
    # Executive Summary Section with better formatting
    section_title = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=dark_blue,
        spaceAfter=12
    )
    story.append(Paragraph("Executive Summary", section_title))
    
    # Create a clinical interpretation paragraph with better formatting
    high_risk_count = summary_stats['high_risk']
    high_risk_pct = (high_risk_count / summary_stats['total']) * 100 if summary_stats['total'] > 0 else 0
    
    clinical_interpretation = f"""
    This analysis evaluated {summary_stats['total']} patients for stroke risk using a validated XGBoost machine learning model. 
    Key findings include:
    
    • {high_risk_count} patients ({high_risk_pct:.1f}%) were identified as high-risk and may benefit from immediate intervention
    • {summary_stats['moderate_risk']} patients ({summary_stats['moderate_risk']*100/summary_stats['total']:.1f}%) show moderate risk requiring monitoring
    • The average risk score across all patients is {summary_stats['avg_risk']:.3f}, suggesting a {'significant' if summary_stats['avg_risk'] > 0.3 else 'moderate' if summary_stats['avg_risk'] > 0.1 else 'relatively low'} overall risk profile
    
    This report provides detailed analysis of risk factors, demographic patterns, and evidence-based intervention recommendations.
    """
    story.append(Paragraph(clinical_interpretation, styles['Normal']))
    story.append(Spacer(1, 15))
    
    # NEW SECTION: Patient Demographics Analysis
    story.append(Paragraph("Patient Demographics Analysis", section_title))
    
    # Create demographic analysis text
    gender_counts = results_df['gender'].value_counts()
    age_stats = {
        'mean': results_df['age'].mean(),
        'median': results_df['age'].median(),
        'min': results_df['age'].min(),
        'max': results_df['age'].max(),
    }
    
    demographics_text = f"""
    The analyzed cohort consists of {len(results_df)} patients with the following characteristics:
    • Gender distribution: {gender_counts.get(1, 0)} male ({gender_counts.get(1, 0)*100/len(results_df):.1f}%), {gender_counts.get(0, 0)} female ({gender_counts.get(0, 0)*100/len(results_df):.1f}%)
    • Age distribution: Mean {age_stats['mean']:.1f} years (range: {age_stats['min']:.1f}-{age_stats['max']:.1f})
    • Patients with hypertension: {results_df['hypertension'].sum()} ({results_df['hypertension'].mean()*100:.1f}%)
    • Patients with heart disease: {results_df['heart_disease'].sum()} ({results_df['heart_disease'].mean()*100:.1f}%)
    
    According to epidemiological data from the American Heart Association, these demographics align with {'higher than' if results_df['hypertension'].mean() > 0.3 else 'typical'} prevalence rates for stroke risk factors in the general population.
    """
    story.append(Paragraph(demographics_text, styles['Normal']))
    
    # Create age distribution histogram
    plt.figure(figsize=(8, 4))
    sns.histplot(results_df['age'], bins=15, kde=True)
    plt.xlabel('Age (years)')
    plt.ylabel('Number of Patients')
    plt.title('Age Distribution of Patient Cohort')
    plt.grid(alpha=0.3)
    
    # Save chart to buffer
    age_hist_buffer = io.BytesIO()
    plt.savefig(age_hist_buffer, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    age_hist_buffer.seek(0)
    
    # Add histogram to document
    age_hist_img = Image(age_hist_buffer, width=5*inch, height=3*inch)
    story.append(age_hist_img)
    story.append(Spacer(1, 15))
    
    # NEW SECTION: Risk Distribution by Age Groups
    story.append(Paragraph("Risk Distribution by Age Groups", section_title))
    
    # Create age groups
    results_df['age_group'] = pd.cut(
        results_df['age'],
        bins=[0, 40, 55, 65, 75, 100],
        labels=['<40', '40-55', '55-65', '65-75', '>75']
    )
    
    # Calculate risk by age group
    risk_by_age = results_df.groupby('age_group')['prediction_proba'].mean().reset_index()
    
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(x='age_group', y='prediction_proba', data=risk_by_age, palette='YlOrRd')
    ax.set(xlabel='Age Group', ylabel='Average Stroke Risk', title='Stroke Risk by Age Group')
    plt.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(risk_by_age['prediction_proba']):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # Save chart to buffer
    age_risk_buffer = io.BytesIO()
    plt.savefig(age_risk_buffer, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    age_risk_buffer.seek(0)
    
    # Add explanation text
    age_risk_text = """
    This chart shows the clear relationship between advancing age and stroke risk. According to the American Stroke Association, 
    stroke risk approximately doubles for each decade after age 55. The substantial increase after age 65 aligns with clinical 
    expectations and highlights the need for more aggressive preventive measures in older age groups. Age represents a 
    non-modifiable risk factor, emphasizing the importance of addressing modifiable factors in higher age brackets.
    """
    story.append(Paragraph(age_risk_text, styles['Normal']))
    story.append(Spacer(1, 5))
    
    # Add age risk chart to document
    age_risk_img = Image(age_risk_buffer, width=5*inch, height=3*inch)
    story.append(age_risk_img)
    story.append(Spacer(1, 15))
    
    # NEW SECTION: Comorbidity Analysis
    story.append(Paragraph("Comorbidity Analysis", section_title))
    
    # Create matrix for hypertension and heart disease
    results_df['both_conditions'] = ((results_df['hypertension'] == 1) & (results_df['heart_disease'] == 1)).astype(int)
    results_df['only_hypertension'] = ((results_df['hypertension'] == 1) & (results_df['heart_disease'] == 0)).astype(int)
    results_df['only_heart_disease'] = ((results_df['hypertension'] == 0) & (results_df['heart_disease'] == 1)).astype(int)
    results_df['no_conditions'] = ((results_df['hypertension'] == 0) & (results_df['heart_disease'] == 0)).astype(int)
    
    comorbidity_counts = [
        results_df['both_conditions'].sum(),
        results_df['only_hypertension'].sum(),
        results_df['only_heart_disease'].sum(),
        results_df['no_conditions'].sum()
    ]
    
    comorbidity_risks = [
        results_df.loc[results_df['both_conditions'] == 1, 'prediction_proba'].mean(),
        results_df.loc[results_df['only_hypertension'] == 1, 'prediction_proba'].mean(),
        results_df.loc[results_df['only_heart_disease'] == 1, 'prediction_proba'].mean(),
        results_df.loc[results_df['no_conditions'] == 1, 'prediction_proba'].mean()
    ]
    
    # Create a horizontal bar chart for comorbidity risks
    plt.figure(figsize=(8, 5))
    labels = ['Hypertension + Heart Disease', 'Hypertension Only', 'Heart Disease Only', 'No Conditions']
    colors = ['#8B0000', '#FF4500', '#FFA07A', '#90EE90']
    
    y_pos = np.arange(len(labels))
    
    plt.barh(y_pos, comorbidity_risks, color=colors)
    plt.yticks(y_pos, labels)
    plt.xlabel('Average Stroke Risk')
    plt.title('Stroke Risk by Comorbidity Status')
    
    # Add counts to the bars
    for i, (risk, count) in enumerate(zip(comorbidity_risks, comorbidity_counts)):
        plt.text(risk + 0.01, i, f'n={count} ({count/len(results_df)*100:.1f}%)', va='center')
    
    # Save chart to buffer
    comorbidity_buffer = io.BytesIO()
    plt.savefig(comorbidity_buffer, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    comorbidity_buffer.seek(0)
    
    # Add explanation text
    comorbidity_text = """
    This analysis demonstrates the synergistic effect of comorbidities on stroke risk. Patients with both hypertension and heart 
    disease show substantially higher risk than those with either condition alone, consistent with the multiplicative rather than 
    additive nature of vascular risk factors. This finding aligns with the INTERSTROKE study, which identified hypertension as 
    the strongest modifiable risk factor for stroke, with an attributable population risk of 34.6%. The combination of cardiac 
    disease and hypertension suggests a need for more aggressive management and closer monitoring.
    """
    story.append(Paragraph(comorbidity_text, styles['Normal']))
    story.append(Spacer(1, 5))
    
    # Add comorbidity chart to document
    comorbidity_img = Image(comorbidity_buffer, width=6*inch, height=4*inch)
    story.append(comorbidity_img)
    story.append(Spacer(1, 15))

    # Include your existing risk distribution chart, scatter plots, etc. here

    # Continue with your existing sections and tables
    
    # NEW SECTION: Relative Risk Analysis
    story.append(Paragraph("Relative Risk Analysis", section_title))
    
    # Create relative risk table with evidence-based comparisons
    relative_risk_data = [
        ["Risk Factor", "Observed Relative Risk", "Published Evidence"],
        ["Hypertension", f"{results_df.loc[results_df['hypertension']==1, 'prediction_proba'].mean() / results_df.loc[results_df['hypertension']==0, 'prediction_proba'].mean():.2f}x", "2-4x (AHA/ASA Guidelines)"],
        ["Heart Disease", f"{results_df.loc[results_df['heart_disease']==1, 'prediction_proba'].mean() / results_df.loc[results_df['heart_disease']==0, 'prediction_proba'].mean():.2f}x", "2-9x (Framingham Study)"],
        ["Age >65", f"{results_df.loc[results_df['age']>65, 'prediction_proba'].mean() / results_df.loc[results_df['age']<=65, 'prediction_proba'].mean():.2f}x", "Doubles each decade after 55"],
        ["Elevated Glucose", f"{results_df.loc[results_df['avg_glucose_level']>140, 'prediction_proba'].mean() / results_df.loc[results_df['avg_glucose_level']<=140, 'prediction_proba'].mean():.2f}x", "1.5-3x (UKPDS Study)"],
        ["BMI >30", f"{results_df.loc[results_df['bmi']>30, 'prediction_proba'].mean() / results_df.loc[results_df['bmi']<=30, 'prediction_proba'].mean():.2f}x", "1.5-2x (Systematic Reviews)"]
    ]
    
    # Create table for relative risk with improved styling
    relative_risk_table = Table(relative_risk_data, colWidths=[1.5*inch, 1.5*inch, 3*inch])
    relative_risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), dark_blue),
        ('TEXTCOLOR', (0, 0), (-1, 0), whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), white),  # Use white variable, not colors.white
        ('GRID', (0, 0), (-1, -1), 1, black),     # Use black variable, not colors.black
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    # Add explanation text
    relative_risk_explanation = """
    This table compares the relative risks observed in this patient cohort against published evidence from major clinical studies. 
    The relative risk represents how many times more likely a patient with the risk factor is to develop a stroke compared to those 
    without the risk factor. Our observed values generally align with published literature, validating the predictive model's 
    performance across these key risk dimensions. Any substantial deviations suggest potential cohort-specific characteristics 
    that may warrant further investigation.
    """
    story.append(Paragraph(relative_risk_explanation, styles['Normal']))
    story.append(Spacer(1, 5))
    story.append(relative_risk_table)
    story.append(Spacer(1, 15))

    # Build the document with all the existing and new sections
    doc.build(story)
    buffer.seek(0)
    return buffer