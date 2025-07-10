"""
Configuration settings for the Stroke Risk Prediction application.
"""

# Model paths
XGBOOST_MODEL_PATH = "models/xgboost_stroke_model.joblib"
MODEL_PATH = XGBOOST_MODEL_PATH  # Alias for backward compatibility

# Data processing configuration
EXPECTED_COLUMNS = [
    'age', 'gender', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
]

REQUIRED_COLUMNS = EXPECTED_COLUMNS

# Categorical variable mappings
GENDER_MAP = {'Male': 1, 'Female': 0, 'Other': 2}
EVER_MARRIED_MAP = {'Yes': 1, 'No': 0}
WORK_TYPE_MAP = {
    'Private': 0, 
    'Self-employed': 1, 
    'Govt_job': 2, 
    'children': 3, 
    'Never_worked': 4
}
RESIDENCE_TYPE_MAP = {'Urban': 1, 'Rural': 0}
SMOKING_STATUS_MAP = {
    'never smoked': 0,
    'formerly smoked': 1, 
    'smokes': 2,
    'Unknown': 3
}

# Risk categories
RISK_CATEGORIES = {
    'Low Risk': (0.0, 0.3),
    'Moderate Risk': (0.3, 0.7),
    'High Risk': (0.7, 1.0)
}

# Gemini API key - explicitly defined
GEMINI_API_KEY = "AIzaSyAWU8UUca9pn0Cv_IsKBMJiPCtk9WdDKAU"

# Sample values for demo purposes
SAMPLE_VALUES = {
    "age": 67,
    "gender": "Male",
    "hypertension": 1,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
}

# Chatbot system prompt
CHATBOT_SYSTEM_PROMPT = """
You are a helpful assistant specialized in stroke risk information for a medical application. 
Your role is to provide educational information about stroke risk factors, prevention, and 
help users understand their risk assessment results.

Guidelines:
1. Provide factual, evidence-based information about stroke risks and prevention
2. Always clarify that you're not providing medical advice and users should consult healthcare professionals
3. Be empathetic but professional when discussing health concerns
4. If asked about topics outside your expertise (like specific treatment plans, diagnoses, or 
   non-stroke related health issues), politely redirect the conversation
5. Avoid making definitive predictions about a user's health outcomes
6. Never contradict or question the model's risk assessment results
7. When discussing lifestyle changes, focus on generally accepted medical guidelines
8. Keep responses concise but informative

Remember, your purpose is educational and supportive, not diagnostic.
"""