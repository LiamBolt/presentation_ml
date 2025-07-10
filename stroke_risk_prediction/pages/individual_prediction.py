"""
Individual Stroke Risk Prediction page for the Stroke Risk Prediction application.
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional

from src.config import SAMPLE_VALUES
from src.utils import load_model, preprocess_input, get_risk_category, format_patient_data, validate_input_data
from src.chatbot import display_chatbot_interface

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_gemini():
    """
    Initialize the Gemini API client.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        api_key = GEMINI_API_KEY
        if not api_key:
            api_key = os.environ.get('GEMINI_API_KEY')
        
        if not api_key:
            logger.error("Gemini API key not found")
            return False
        
        genai.configure(api_key=api_key)
        logger.info("Gemini API initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing Gemini API: {str(e)}")
        return False

def create_chat_session(
    user_data: Optional[Dict[str, Any]] = None, 
    prediction: Optional[float] = None
):
    """
    Create a new chat session with context about the user's data and prediction.
    
    Args:
        user_data (Dict[str, Any], optional): User input data
        prediction (float, optional): Prediction probability
        
    Returns:
        genai.ChatSession: Chat session object or None if initialization fails
    """
    try:
        if not initialize_gemini():
            return None
        
        # Create a message with context if user data is available
        context_message = ""
        if user_data and prediction is not None:
            risk_level = "high" if prediction > 0.5 else "low"
            context_message = f"""
            I have a user who has received a stroke risk prediction from our XGBoost model.
            User details: {json.dumps(user_data, indent=2)}
            Prediction: {risk_level} risk ({prediction:.1%} probability)
            
            Please provide helpful responses based on this context, but remember:
            1. Don't contradict the model's prediction
            2. Always clarify you're not providing medical advice
            3. Focus on general stroke risk information and preventive measures
            """
        
        # Initialize the model
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
            system_instruction=CHATBOT_SYSTEM_PROMPT + "\n\n" + context_message
        )
        
        # Create chat session
        chat = model.start_chat(history=[])
        return chat
        
    except Exception as e:
        logger.error(f"Error creating chat session: {str(e)}")
        return None

def get_chatbot_response(
    chat_session, 
    user_query: str
) -> str:
    """
    Get a response from the chatbot for a user query.
    
    Args:
        chat_session: Active chat session
        user_query (str): User's question or message
        
    Returns:
        str: Chatbot's response or error message
    """
    try:
        if not chat_session:
            return "I'm sorry, but the chatbot service is currently unavailable. Please try again later."
        
        # Handle common out-of-scope queries
        if any(keyword in user_query.lower() for keyword in ['loan', 'credit', 'finance', 'money', 'invest']):
            return "I'm specialized in stroke risk information and can't provide financial advice. Is there anything about stroke risk factors or prevention you'd like to know?"
        
        # Get response from Gemini
        response = chat_session.send_message(user_query)
        return response.text
        
    except Exception as e:
        logger.error(f"Error getting chatbot response: {str(e)}")
        return f"I'm sorry, I encountered an error while processing your request. Please try asking again or rephrase your question."

def app():
    """Main function for the Individual Prediction page."""
    st.title("Individual Stroke Risk Assessment")
    st.write("Enter patient information to predict individual stroke risk.")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("⚠️ Error: Unable to load the prediction model.")
        return
    
    # Create two columns: form and results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Patient Information")
        
        # Sample data toggle
        use_sample_data = st.checkbox("Use Sample Data", value=False)
        
        # Create input form
        with st.form("patient_form"):
            if use_sample_data:
                age = st.number_input("Age", min_value=0, max_value=120, value=SAMPLE_VALUES["age"], step=1)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0 if SAMPLE_VALUES["gender"] == "Male" else 1)
                hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=SAMPLE_VALUES["hypertension"])
                heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=SAMPLE_VALUES["heart_disease"])
                ever_married = st.selectbox("Ever Married", ["Yes", "No"], index=0 if SAMPLE_VALUES["ever_married"] == "Yes" else 1)
                work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"], index=0)
                residence_type = st.selectbox("Residence Type", ["Urban", "Rural"], index=0 if SAMPLE_VALUES["Residence_type"] == "Urban" else 1)
                avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=0.0, max_value=500.0, value=SAMPLE_VALUES["avg_glucose_level"], step=0.1)
                bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=SAMPLE_VALUES["bmi"], step=0.1)
                smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"], index=1)
            else:
                age = st.number_input("Age", min_value=0, max_value=120, value=45, step=1)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                ever_married = st.selectbox("Ever Married", ["Yes", "No"])
                work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
                residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
                avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=0.0, max_value=500.0, value=85.0, step=0.1)
                bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
                smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
            
            submit_prediction = st.form_submit_button("Predict Risk", use_container_width=True, type="primary")
    
    with col2:
        st.subheader("Prediction Results")
        
        # Initialize prediction variables
        prediction_made = False
        prediction_proba = None
        risk_category = None
        user_data = None
        
        if submit_prediction:
            # Collect input data
            user_data = {
                "age": age,
                "gender": gender,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "ever_married": ever_married,
                "work_type": work_type,
                "Residence_type": residence_type,
                "avg_glucose_level": avg_glucose_level,
                "bmi": bmi,
                "smoking_status": smoking_status
            }
            
            # Validate input data
            is_valid, error_message = validate_input_data(user_data)
            
            if not is_valid:
                st.error(f"Input validation error: {error_message}")
            else:
                try:
                    with st.spinner("Making prediction..."):
                        # Preprocess the data
                        X = preprocess_input(user_data)
                        prediction     = model.predict(X)[0]
                        prediction_proba = model.predict_proba(X)[0,1]
                        
                        # Get risk category
                        risk_category, risk_color, risk_description = get_risk_category(prediction_proba)
                        
                        prediction_made = True
                        
                        # Display results
                        st.success("Prediction completed successfully!")
                        
                        # Display risk probability
                        st.metric(
                            label="Stroke Risk Probability",
                            value=f"{prediction_proba:.1%}",
                            delta=None
                        )
                        
                        # Display risk category with color
                        if risk_color == "green":
                            st.success(f"**Risk Category:** {risk_category}")
                        elif risk_color == "orange":
                            st.warning(f"**Risk Category:** {risk_category}")
                        elif risk_color == "red":
                            st.error(f"**Risk Category:** {risk_category}")
                        
                        st.info(f"**Description:** {risk_description}")
                        
                        # Display formatted patient data
                        st.subheader("Patient Summary")
                        formatted_data = format_patient_data(user_data)
                        
                        for key, value in formatted_data.items():
                            st.text(f"{key}: {value}")
                        
                        # Recommendations based on risk level
                        st.subheader("Recommendations")
                        
                        if risk_category == "Low Risk":
                            st.success("""
                            **Low Risk Recommendations:**
                            - Continue maintaining a healthy lifestyle
                            - Regular check-ups with your healthcare provider
                            - Monitor blood pressure and glucose levels
                            - Stay physically active
                            """)
                        elif risk_category == "Moderate Risk":
                            st.warning("""
                            **Moderate Risk Recommendations:**
                            - Consult with your healthcare provider for personalized advice
                            - Consider lifestyle modifications (diet, exercise)
                            - Monitor and manage risk factors closely
                            - Regular medical check-ups
                            """)
                        elif risk_category == "High Risk":
                            st.error("""
                            **High Risk Recommendations:**
                            - **Consult with a healthcare provider immediately**
                            - Discuss preventive measures and treatment options
                            - Close monitoring of all risk factors
                            - Consider medication if recommended by your doctor
                            - Lifestyle changes are crucial
                            """)
                        
                        # Store data in session state for chatbot
                        st.session_state['last_prediction_data'] = user_data
                        st.session_state['last_prediction_proba'] = prediction_proba
                        
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.info("Please check your input data and try again.")
        
        else:
            st.info("👈 Enter patient information and click 'Predict Risk' to see results.")
    
    # Chatbot section (full width) - This should make the chatbot visible
    st.divider()
    st.subheader("🤖 Ask Questions About Stroke Risk")
    
    # Display chatbot interface
    user_data_for_chat = st.session_state.get('last_prediction_data', None)
    prediction_for_chat = st.session_state.get('last_prediction_proba', None)
    
    try:
        display_chatbot_interface(user_data_for_chat, prediction_for_chat)
    except Exception as e:
        st.error(f"Chatbot unavailable: {str(e)}")
        st.info("The chatbot feature is currently unavailable. Please check your API configuration.")

if __name__ == "__main__":
    app()
