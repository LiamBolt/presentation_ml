import streamlit as st

def app():
    """Main function for the About page."""
    st.title("About Stroke Risk Prediction")
    
    st.write("This application provides a user-friendly interface to predict stroke risk using machine learning.")
    
    # Create tabs for different sections
    tabs = st.tabs([
        "Application Overview",
        "Dataset Information", 
        "Model Details", 
        "Important Disclaimer", 
        "References"
    ])
    
    # Application Overview Tab
    with tabs[0]:
        st.subheader("Purpose & Features")
        
        st.markdown("""
        This application uses an XGBoost machine learning model to predict the likelihood of stroke based on patient health data.
        
        ### Key Features:
        
        - **Individual Risk Assessment**: Predict stroke risk for individual patients
        - **Batch Prediction**: Process multiple patient records at once (for medical professionals)
        - **Model Performance Analysis**: Explore the performance metrics of the model
        - **AI Assistant**: Chat with an AI assistant to understand your results and learn about stroke risk factors
        - **Report Generation**: Download detailed PDF reports of risk assessments
        
        ### How It Works:
        
        1. **Enter patient information** including demographics, health indicators, and lifestyle factors
        2. **Get an immediate risk assessment** with visualizations and explanations
        3. **Explore contributing factors** to understand key risk drivers
        4. **Generate a detailed report** for record-keeping or sharing with healthcare providers
        """)
        
        st.info("This application is designed for educational and research purposes only and should not replace professional medical advice.")
    
    # Dataset Information Tab
    with tabs[1]:
        st.subheader("Dataset Information")
        
        st.markdown("""
        The model was trained on the Healthcare Stroke Prediction Dataset, which contains the following features:
        
        ### Demographics
        - **age**: Age of the patient
        - **gender**: Gender of the patient (Male, Female, Other)
        
        ### Medical History
        - **hypertension**: Whether the patient has hypertension (0: No, 1: Yes)
        - **heart_disease**: Whether the patient has heart disease (0: No, 1: Yes)
        - **ever_married**: Whether the patient is married (Yes, No)
        
        ### Socioeconomic Factors
        - **work_type**: Type of employment (Private, Self-employed, Govt_job, children, Never_worked)
        - **Residence_type**: Type of residence area (Urban, Rural)
        
        ### Health Indicators
        - **avg_glucose_level**: Average glucose level in blood (mg/dL)
        - **bmi**: Body Mass Index
        - **smoking_status**: Smoking status (formerly smoked, never smoked, smokes, Unknown)
        
        ### Target Variable
        - **stroke**: Whether the patient had a stroke (0: No, 1: Yes)
        """)
        
        st.markdown("""
        ### Dataset Characteristics
        
        - **Number of Instances**: 5,110
        - **Class Distribution**: Imbalanced (stroke cases: ~4.9%)
        - **Feature Types**: Numerical and categorical
        
        To address class imbalance, the SMOTEENN technique (combination of SMOTE and Edited Nearest Neighbors) 
        was applied during model training to generate synthetic samples of the minority class and clean the 
        resulting distribution.
        """)
    
    # Model Details Tab
    with tabs[2]:
        st.subheader("XGBoost Model Information")
        
        st.markdown("""
        This application uses the XGBoost algorithm, which was selected after comparative analysis with other models.
        
        ### XGBoost (eXtreme Gradient Boosting)
        
        XGBoost is an optimized gradient boosting algorithm that uses decision trees as base learners. It builds trees
        sequentially, with each tree correcting the errors of the previous ones.
        
        ### Why XGBoost for Stroke Prediction?
        
        - **Handles imbalanced medical data** effectively
        - **Captures complex non-linear relationships** between risk factors
        - **Provides feature importance** insights
        - **High predictive performance** compared to alternative models
        - **Robust to overfitting** through regularization
        
        ### Model Training Process
        
        1. **Data preprocessing**: Handling missing values, encoding categorical features
        2. **Class imbalance handling**: Using SMOTEENN to balance the dataset
        3. **Feature selection**: Based on medical relevance and statistical importance
        4. **Hyperparameter tuning**: Grid search with cross-validation
        5. **Model evaluation**: Using metrics appropriate for imbalanced medical data
        
        ### Performance Metrics
        
        The XGBoost model achieves:
        
        - Accuracy: 92%
        - Precision: 88%
        - Recall: 91%
        - F1 Score: 89%
        - AUC: 95.8%
        """)
    
    # Disclaimer Tab
    with tabs[3]:
        st.subheader("Important Medical Disclaimer")
        
        st.warning("""
        ### Educational Use Only
        
        This application is for **educational and research purposes only**. It is not intended to be a substitute for 
        professional medical advice, diagnosis, or treatment.
        
        ### Not a Medical Device
        
        The stroke risk prediction provided by this application is based on statistical patterns in historical data 
        and should not be considered as clinical advice or diagnosis.
        
        ### Consult Healthcare Professionals
        
        Always consult with qualified healthcare professionals for:
        - Medical advice
        - Diagnosis of health conditions
        - Treatment recommendations
        - Questions about your personal health situation
        
        ### Limitations
        
        The predictions have several limitations:
        - Based on limited demographic and health data
        - May not account for all possible stroke risk factors
        - Cannot capture unique individual circumstances
        - Model accuracy is not 100%
        
        ### Decision-Making
        
        Do not make medical decisions based solely on the results provided by this application. The application's 
        developers, contributors, and affiliated organizations are not responsible for any healthcare decisions 
        made based on the application's output.
        """)
    
    # References Tab
    with tabs[4]:
        st.subheader("References & Acknowledgments")
        
        st.markdown("""
        ### Dataset
        
        The Healthcare Stroke Prediction Dataset used for training the models in this application.
        
        ### Machine Learning Libraries
        
        - XGBoost: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
        - Scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)
        - imbalanced-learn (SMOTEENN): [https://imbalanced-learn.org/](https://imbalanced-learn.org/)
        
        ### Visualization Libraries
        
        - Plotly: [https://plotly.com/](https://plotly.com/)
        - Matplotlib & Seaborn: For notebook visualizations
        
        ### Application Framework
        
        - Streamlit: [https://streamlit.io/](https://streamlit.io/)
        
        ### Medical References
        
        - American Stroke Association: [https://www.stroke.org/](https://www.stroke.org/)
        - World Health Organization - Stroke information: [https://www.who.int/health-topics/stroke](https://www.who.int/health-topics/stroke)
        """)
        
        st.markdown("""
        ### Acknowledgments
        
        - Thanks to the open-source community for the tools and libraries that made this project possible
        - Special thanks to healthcare professionals who provided domain knowledge guidance
        
        ### License
        
        This project is licensed under the MIT License.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div class="footer">
            <p>© 2025 Stroke Risk Prediction | Developed for educational purposes</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    app()