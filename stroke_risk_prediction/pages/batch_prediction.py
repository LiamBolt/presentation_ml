import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import time
import os
import joblib

# Import from src
from src.utils import load_model, get_risk_category, create_report_pdf, create_download_link
from src.data_processor import preprocess_dataset, generate_enhanced_medical_report
from src.config import MODEL_PATH

# Add this function to get model's expected feature order
def get_model_features():
    model = joblib.load(MODEL_PATH)
    return list(model.feature_names_in_)

def app():
    """Main function for the Batch Prediction page for medical experts."""
    st.title("Batch Stroke Risk Assessment")
    st.write("Upload a dataset to predict stroke risk for multiple patients.")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("⚠️ Error: Unable to load the prediction model.")
        return
    
    # File upload section
    st.subheader("Upload Patient Data")
    
    # Example file template
    st.info("""
    ### File Format Requirements
    
    Upload a CSV or Excel file with the following columns:
    - age: Patient's age (numeric)
    - gender: 'Male', 'Female', or 'Other'
    - hypertension: 0 (no) or 1 (yes)
    - heart_disease: 0 (no) or 1 (yes)
    - ever_married: 'Yes' or 'No'
    - work_type: 'Private', 'Self-employed', 'Govt_job', 'children', or 'Never_worked'
    - Residence_type: 'Urban' or 'Rural'
    - avg_glucose_level: Average glucose level in blood (numeric)
    - bmi: Body Mass Index (numeric)
    - smoking_status: 'formerly smoked', 'never smoked', 'smokes', or 'Unknown'
    
    The column order doesn't matter - our system will handle it automatically.
    """)
    
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            file_extension = uploaded_file.name.split(".")[-1]
            
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_extension in ["xlsx", "xls"]:
                try:
                    df = pd.read_excel(uploaded_file)
                except ImportError:
                    st.error("Excel support package missing. Please run: pip install openpyxl")
                    st.info("Alternatively, save your file as CSV and upload again.")
                    return
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                return
            
            # Display raw data
            st.subheader("Raw Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Define the required columns (but don't enforce order)
            required_columns = [
                'age', 'gender', 'hypertension', 'heart_disease', 'ever_married',
                'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
            ]
            
            # Check if all required columns are present (ignoring order)
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.info("Please check the file format requirements and upload a valid file.")
                return
            
            # Process the data
            if st.button("Generate Predictions", use_container_width=True):
                with st.spinner("Processing data and generating predictions..."):
                    # Create a copy of the dataframe
                    results_df = df.copy()
                    
                    # Initialize prediction columns
                    results_df["prediction"] = None
                    results_df["prediction_proba"] = None
                    results_df["risk_category"] = None
                    
                    # Load the model
                    model = load_model()
                    
                    # Use the whole dataset approach rather than row-by-row for better performance
                    try:
                        # First check if all crucial columns are present
                        crucial_columns = ['age', 'gender', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
                        missing_crucial = [col for col in crucial_columns if col not in df.columns]
                        
                        if missing_crucial:
                            st.error(f"Missing crucial columns: {', '.join(missing_crucial)}")
                            st.info("These columns are required for prediction.")
                            return
                        
                        # Process the entire dataset at once
                        X = preprocess_dataset(df)
                        preds = model.predict(X)
                        probas = model.predict_proba(X)[:,1]
                        
                        # Get risk categories for all predictions
                        risk_categories = []
                        risk_colors = []
                        risk_descriptions = []
                        for prob in probas:
                            category, color, description = get_risk_category(prob)
                            risk_categories.append(category)
                            risk_colors.append(color)
                            risk_descriptions.append(description)
                        
                        # Update results dataframe
                        results_df["prediction"] = preds
                        results_df["prediction_proba"] = probas
                        results_df["risk_category"] = risk_categories
                        results_df["risk_color"] = risk_colors
                        results_df["risk_description"] = risk_descriptions
                        
                    except Exception as e:
                        error_message = str(e)
                        if "feature_names mismatch" in error_message:
                            st.error(f"Feature order mismatch error. Processing rows individually.")
                            
                            # Fall back to row-by-row processing
                            for idx, row in df.iterrows():
                                try:
                                    # Create input data dictionary
                                    input_data = row.to_dict()
                                    
                                    # Process only crucial columns if others are missing
                                    crucial_columns = ['age', 'gender', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
                                    missing_crucial = [col for col in crucial_columns if col not in input_data]
                                    
                                    if missing_crucial:
                                        st.warning(f"Row {idx + 1} is missing crucial columns: {', '.join(missing_crucial)}. Skipping.")
                                        continue
                                    
                                    # Create a single-row dataframe for processing
                                    single_row_df = pd.DataFrame([input_data])
                                    
                                    # Ensure columns are in the correct order for the model
                                    X = preprocess_dataset(single_row_df)
                                    
                                    # Make prediction
                                    prediction = model.predict(X)[0]
                                    prediction_proba = model.predict_proba(X)[0][1]
                                    
                                    # Get risk category
                                    risk_category, _, _ = get_risk_category(prediction_proba)
                                    
                                    # Update results dataframe
                                    results_df.at[idx, "prediction"] = prediction
                                    results_df.at[idx, "prediction_proba"] = prediction_proba
                                    results_df.at[idx, "risk_category"] = risk_category
                                    
                                except Exception as e:
                                    st.error(f"Error processing row {idx + 1}: {str(e)}")
                                    # Set default values for failed predictions
                                    results_df.at[idx, "prediction"] = 0
                                    results_df.at[idx, "prediction_proba"] = 0.0
                                    results_df.at[idx, "risk_category"] = "Error"
                        else:
                            st.error(f"Error processing dataset: {error_message}")
                            return
                    
                    # Format the probability column for display
                    results_df["formatted_proba"] = results_df["prediction_proba"].map("{:.2%}".format)
                    
                    # Display the results
                    st.subheader("Prediction Results")
                    display_cols = df.columns.tolist() + ["prediction", "formatted_proba", "risk_category"]
                    st.dataframe(results_df[display_cols].rename(columns={"formatted_proba": "risk_probability"}), use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("Summary Statistics")
                    
                    # Calculate summary statistics
                    total_patients = len(results_df)
                    high_risk_count = sum(results_df["risk_category"] == "High Risk")
                    medium_risk_count = sum(results_df["risk_category"] == "Moderate Risk")
                    low_risk_count = sum(results_df["risk_category"] == "Low Risk")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Patients", total_patients)
                    
                    with col2:
                        st.metric("High Risk", high_risk_count)
                    
                    with col3:
                        st.metric("Moderate Risk", medium_risk_count)
                    
                    with col4:
                        st.metric("Low Risk", low_risk_count)
                    
                    # Risk distribution visualization
                    st.subheader("Risk Distribution")
                    
                    # Create risk distribution chart
                    risk_counts = results_df["risk_category"].value_counts().reset_index()
                    risk_counts.columns = ["Risk Category", "Count"]
                    
                    # Ensure all risk categories are present
                    all_categories = ["Low Risk", "Moderate Risk", "High Risk"]
                    for category in all_categories:
                        if category not in risk_counts["Risk Category"].values:
                            risk_counts = pd.concat([
                                risk_counts,
                                pd.DataFrame({"Risk Category": [category], "Count": [0]})
                            ])
                    
                    # Create color map
                    color_map = {
                        "Low Risk": "green",
                        "Moderate Risk": "orange",
                        "High Risk": "red"
                    }
                    
                    # Create pie chart
                    fig = px.pie(
                        risk_counts,
                        values="Count",
                        names="Risk Category",
                        title="Risk Distribution",
                        color="Risk Category",
                        color_discrete_map=color_map
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature analysis
                    st.subheader("Feature Analysis")
                    
                    # Age distribution by risk category
                    fig_age = px.box(
                        results_df,
                        x="risk_category",
                        y="age",
                        color="risk_category",
                        title="Age Distribution by Risk Category",
                        color_discrete_map=color_map
                    )
                    
                    st.plotly_chart(fig_age, use_container_width=True)
                    
                    # Glucose level distribution by risk category
                    fig_glucose = px.box(
                        results_df,
                        x="risk_category",
                        y="avg_glucose_level",
                        color="risk_category",
                        title="Glucose Level Distribution by Risk Category",
                        color_discrete_map=color_map
                    )
                    
                    st.plotly_chart(fig_glucose, use_container_width=True)
                    
                    # Download options
                    st.subheader("Download Options")
                    
                    # Create Excel file
                    excel_buffer = BytesIO()
                    results_df.to_excel(excel_buffer, index=False)
                    excel_buffer.seek(0)
                    
                    # Create download link for Excel
                    excel_b64 = base64.b64encode(excel_buffer.read()).decode()
                    excel_href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{excel_b64}" download="stroke_risk_predictions.xlsx" class="download-btn">Download Predictions as Excel</a>'
                    
                    st.markdown(excel_href, unsafe_allow_html=True)
                    
                    # Create comprehensive report
                    st.markdown("### Comprehensive Analysis Report")
                    
                    # Generate comprehensive PDF report
                    with st.spinner("Generating comprehensive report..."):
                        try:
                            from reportlab.lib.pagesizes import letter, landscape
                            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
                            from reportlab.lib import colors
                            from reportlab.lib.styles import getSampleStyleSheet
                            from reportlab.lib.units import inch
                            import matplotlib.pyplot as plt
                            from datetime import datetime
                            import io
                            
                            # Create a BytesIO buffer
                            pdf_buffer = BytesIO()
                            
                            # Create the PDF document
                            doc = SimpleDocTemplate(pdf_buffer, pagesize=landscape(letter))
                            elements = []
                            
                            # Get styles
                            styles = getSampleStyleSheet()
                            title_style = styles["Heading1"]
                            subtitle_style = styles["Heading2"]
                            normal_style = styles["Normal"]
                            
                            # Add title
                            elements.append(Paragraph("Stroke Risk Prediction - Batch Analysis Report", title_style))
                            elements.append(Spacer(1, 0.25*inch))
                            
                            # Add date
                            date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                            elements.append(Paragraph(f"Generated on: {date_str}", normal_style))
                            elements.append(Spacer(1, 0.25*inch))
                            
                            # Add summary statistics
                            elements.append(Paragraph("Summary Statistics", subtitle_style))
                            
                            # Create a table for summary statistics
                            summary_data = [
                                ["Total Patients", "High Risk", "Moderate Risk", "Low Risk"],
                                [str(total_patients), str(high_risk_count), str(medium_risk_count), str(low_risk_count)]
                            ]
                            
                            summary_table = Table(summary_data, colWidths=[2*inch, 2*inch, 2*inch, 2*inch])
                            summary_table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (3, 0), colors.lightgrey),
                                ('TEXTCOLOR', (0, 0), (3, 0), colors.black),
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                            ]))
                            
                            elements.append(summary_table)
                            elements.append(Spacer(1, 0.25*inch))
                            
                            # Add risk distribution chart
                            elements.append(Paragraph("Risk Distribution", subtitle_style))
                            
                            # Create pie chart using matplotlib
                            plt.figure(figsize=(8, 4))
                            plt.pie(
                                risk_counts["Count"],
                                labels=risk_counts["Risk Category"],
                                autopct='%1.1f%%',
                                colors=['green', 'orange', 'red']
                            )
                            plt.title("Risk Distribution")
                            plt.axis('equal')
                            
                            # Save chart to buffer
                            chart_buffer = io.BytesIO()
                            plt.savefig(chart_buffer, format='png')
                            plt.close()
                            chart_buffer.seek(0)
                            
                            # Add chart to PDF
                            chart_image = Image(chart_buffer, width=6*inch, height=3*inch)
                            elements.append(chart_image)
                            elements.append(Spacer(1, 0.25*inch))
                            
                            # Add key findings
                            elements.append(Paragraph("Key Findings", subtitle_style))
                            
                            # Calculate some statistics
                            high_risk_percentage = (high_risk_count / total_patients) * 100 if total_patients > 0 else 0
                            
                            findings = [
                                f"{high_risk_percentage:.1f}% of patients are at high risk of stroke.",
                                f"The average age of high-risk patients is {results_df[results_df['risk_category'] == 'High Risk']['age'].mean():.1f} years.",
                                f"The average glucose level of high-risk patients is {results_df[results_df['risk_category'] == 'High Risk']['avg_glucose_level'].mean():.1f} mg/dL."
                            ]
                            
                            for finding in findings:
                                elements.append(Paragraph(f"• {finding}", normal_style))
                            
                            elements.append(Spacer(1, 0.25*inch))
                            
                            # Add top 10 highest risk patients
                            elements.append(Paragraph("Top 10 Highest Risk Patients", subtitle_style))
                            
                            # Get top 10 patients by prediction probability
                            top10_df = results_df.sort_values(by="prediction_proba", ascending=False).head(10)
                            
                            # Create table data
                            table_data = [["ID", "Age", "Gender", "Hypertension", "Heart Disease", "Glucose", "BMI", "Risk"]]
                            
                            for idx, row in top10_df.iterrows():
                                table_data.append([
                                    str(idx + 1),
                                    str(row["age"]),
                                    str(row["gender"]),
                                    "Yes" if row["hypertension"] == 1 else "No",
                                    "Yes" if row["heart_disease"] == 1 else "No",
                                    str(row["avg_glucose_level"]),
                                    str(row["bmi"]),
                                    str(row["prediction_proba"])
                                ])
                            
                            # Create the table
                            table = Table(table_data)
                            table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                            ]))
                            
                            elements.append(table)
                            elements.append(Spacer(1, 0.25*inch))
                            
                            # Add recommendations
                            elements.append(Paragraph("Recommendations", subtitle_style))
                            
                            recommendations = [
                                "Consider prioritizing follow-up for the high-risk patients identified in this report.",
                                "Monitor glucose levels closely for patients with elevated values, as this is a key risk factor.",
                                "Provide preventive education to all patients, with targeted interventions for those at moderate to high risk.",
                                "Consider additional screening for patients with multiple risk factors even if their overall risk score is moderate."
                            ]
                            
                            for recommendation in recommendations:
                                elements.append(Paragraph(f"• {recommendation}", normal_style))
                            
                            elements.append(Spacer(1, 0.25*inch))
                            
                            # Add disclaimer
                            elements.append(Paragraph("Disclaimer", subtitle_style))
                            elements.append(Paragraph("This report is generated by a machine learning model and is for informational purposes only. It does not constitute medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for proper medical guidance.", normal_style))
                            
                            # Build the PDF
                            doc.build(elements)
                            pdf_buffer.seek(0)
                            
                            # Create download link for PDF
                            pdf_b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
                            pdf_href = f'<a href="data:application/pdf;base64,{pdf_b64}" download="stroke_risk_analysis_report.pdf" class="download-btn">Download Comprehensive Analysis Report (PDF)</a>'
                            
                            st.markdown(pdf_href, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error generating PDF report: {str(e)}")
                    
                    # Enhanced medical expert report
                    st.markdown("### Enhanced Medical Expert Report")
                    with st.spinner("Generating enhanced medical report..."):
                        try:
                            # Create summary stats for report
                            summary_stats = {
                                'total': total_patients,
                                'high_risk': high_risk_count,
                                'moderate_risk': medium_risk_count,
                                'low_risk': low_risk_count,
                                'avg_risk': results_df["prediction_proba"].mean()
                            }
                            
                            # Generate the enhanced medical report
                            medical_report_buffer = generate_enhanced_medical_report(results_df, summary_stats)
                            
                            # Create download link
                            medical_report_b64 = base64.b64encode(medical_report_buffer.getvalue()).decode()
                            medical_report_href = f'<a href="data:application/pdf;base64,{medical_report_b64}" download="enhanced_medical_stroke_report.pdf" class="download-btn">Download Enhanced Medical Report (PDF)</a>'
                            
                            st.markdown(medical_report_href, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error generating enhanced medical report: {str(e)}")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please make sure your file is properly formatted and try again.")

if __name__ == "__main__":
    app()