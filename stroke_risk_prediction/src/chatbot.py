"""
Chatbot functionality using Google's Gemini API for the Stroke Risk Prediction application.
"""

import google.generativeai as genai
import streamlit as st
import logging
from typing import Dict, Any, List, Optional
import json
import os

from src.config import CHATBOT_SYSTEM_PROMPT, GEMINI_API_KEY, SAMPLE_VALUES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_gemini():
    """Initialize the Gemini API client."""
    try:
        # Use the explicit API key from config
        api_key = GEMINI_API_KEY
        
        # Configure Gemini with explicit key
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
            model_name="gemini-2.5-flash-lite-preview-06-17",
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

def display_chatbot_interface(
    user_data: Optional[Dict[str, Any]] = None, 
    prediction: Optional[float] = None
):
    """Display the chatbot interface in the Streamlit app."""
    
    # Initialize session state for chat history if it doesn't exist
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'chat_session' not in st.session_state:
        # Try to create a new chat session
        try:
            st.session_state.chat_session = create_chat_session(user_data, prediction)
            if not st.session_state.chat_session:
                st.warning("Chatbot service is currently unavailable. Using simplified mode.")
        except Exception as e:
            logger.error(f"Error creating chat session: {str(e)}")
            st.warning("Gemini API quota exceeded. Using simplified response mode.")
            st.session_state.chat_session = None
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about stroke risk..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from chatbot
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if st.session_state.chat_session:
                        response = get_chatbot_response(st.session_state.chat_session, prompt)
                    else:
                        # Fallback responses for common questions when API is unavailable
                        response = get_fallback_response(prompt, user_data, prediction)
                    st.markdown(response)
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg and "quota" in error_msg:
                        st.warning("Gemini API quota exceeded. Using simplified responses.")
                        # Switch to fallback mode
                        st.session_state.chat_session = None
                        response = get_fallback_response(prompt, user_data, prediction)
                        st.markdown(response)
                    else:
                        response = f"Sorry, I encountered an error. Please try again later."
                        st.error(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def get_fallback_response(prompt: str, user_data: Optional[Dict[str, Any]] = None, prediction: Optional[float] = None) -> str:
    """
    Generate fallback responses when the API is unavailable.
    
    Args:
        prompt (str): User's question
        user_data (Dict[str, Any], optional): User input data
        prediction (float, optional): Prediction probability
        
    Returns:
        str: Fallback response
    """
    prompt_lower = prompt.lower()
    
    # General disclaimer
    disclaimer = "\n\n*Note: This is a simplified response as our AI service is temporarily unavailable. Please consult a healthcare professional for personalized advice.*"
    
    # Check for common questions
    if any(word in prompt_lower for word in ["risk", "stroke", "chance", "probability", "likelihood"]):
        if prediction is not None:
            risk_level = "high" if prediction > 0.5 else "moderate" if prediction > 0.3 else "low"
            return f"Based on the provided information, your stroke risk appears to be {risk_level} ({prediction:.1%} probability). Key risk factors include age, blood pressure, glucose levels, and smoking status.{disclaimer}"
        else:
            return f"Stroke risk is influenced by factors like age, hypertension, heart disease, diabetes, smoking, and physical inactivity. Consult a healthcare provider for a personalized risk assessment.{disclaimer}"
    
    elif any(word in prompt_lower for word in ["prevent", "prevention", "reduce", "lower", "decrease"]):
        return f"Stroke risk can be reduced by maintaining a healthy lifestyle, including regular exercise, balanced diet, not smoking, limiting alcohol intake, and managing conditions like high blood pressure and diabetes.{disclaimer}"
    
    elif any(word in prompt_lower for word in ["symptom", "sign", "warning"]):
        return f"Common stroke symptoms include sudden numbness/weakness (especially on one side), confusion, trouble speaking/understanding, vision problems, dizziness/loss of balance, and severe headache. Remember FAST: Face drooping, Arm weakness, Speech difficulty, Time to call emergency services.{disclaimer}"
    
    elif any(word in prompt_lower for word in ["treatment", "medicine", "medication", "drug"]):
        return f"Stroke treatments may include clot-busting medications (for ischemic strokes), surgical interventions, and rehabilitation therapy. Prevention medications might include anticoagulants, anti-hypertensives, statins, and anti-diabetic drugs. Always consult a healthcare provider for appropriate treatment.{disclaimer}"
    
    elif any(word in prompt_lower for word in ["diet", "food", "eat", "nutrition"]):
        return f"A heart-healthy diet can help reduce stroke risk. This includes plenty of fruits, vegetables, whole grains, lean proteins, and healthy fats (like olive oil). Limit sodium, saturated fats, processed foods, and added sugars. The Mediterranean and DASH diets are often recommended.{disclaimer}"
    
    elif any(word in prompt_lower for word in ["exercise", "physical", "activity", "workout"]):
        return f"Regular physical activity helps reduce stroke risk. Aim for at least 150 minutes of moderate-intensity exercise per week. Activities like walking, swimming, cycling, and strength training are beneficial. Always start gradually and consult a healthcare provider before beginning a new exercise program.{disclaimer}"
    
    else:
        return f"I understand you're asking about stroke risk, but I'm currently operating in simplified mode due to service limitations. Please ask about stroke risk factors, prevention, symptoms, treatment, diet, or exercise recommendations.{disclaimer}"