import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('diabetes_model.pkl')
        return model, None
    except FileNotFoundError:
        error_msg = "Model file 'diabetes_model.pkl' not found. Please ensure the model file is in the same directory."
        return None, error_msg
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}. This might be due to version compatibility issues."
        return None, error_msg

model, model_error = load_model()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #c62828;
    }
    .risk-moderate {
        background-color: #fff3e0;
        color: #ef6c00;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ef6c00;
    }
    .risk-low {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🩺 Diabetes Risk Prediction System</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
        AI-powered diabetes risk assessment using advanced machine learning techniques
    </p>
    <p style="font-style: italic; color: #888;">
        Based on the Pima Indian Diabetes Database with medical knowledge integration
    </p>
</div>
""", unsafe_allow_html=True)

# Display model status
if model_error:
    st.error(f"⚠️ **Model Loading Issue**: {model_error}")
    st.warning("🔄 **Fallback Mode**: Using rule-based risk assessment system")
    st.info("💡 **Note**: For best results, please ensure model compatibility or retrain the model with current scikit-learn version")
else:
    st.success("✅ **AI Model**: Loaded successfully and ready for predictions")

# Sidebar for input parameters
st.sidebar.markdown('<h2 class="sub-header">📋 Patient Information</h2>', unsafe_allow_html=True)

# Medical context information
with st.sidebar.expander("📚 Medical Reference Values", expanded=False):
    st.markdown("""
    **Glucose (mg/dL):**
    - Normal: < 140
    - Prediabetes: 140-200
    - Diabetes: > 200
    
    **BMI (kg/m²):**
    - Underweight: < 18.5
    - Normal: 18.5-24.9
    - Overweight: 25.0-29.9
    - Obese: ≥ 30.0
    
    **Blood Pressure (mm Hg):**
    - Normal: < 80
    - Stage 1 HTN: 80-89
    - Stage 2 HTN: ≥ 90
    """)

# Input fields
pregnancies = st.sidebar.number_input(
    "Number of Pregnancies", 
    min_value=0, max_value=20, value=1,
    help="Total number of pregnancies"
)

glucose = st.sidebar.number_input(
    "Glucose Level (mg/dL)", 
    min_value=50.0, max_value=300.0, value=120.0, step=1.0,
    help="Plasma glucose concentration from oral glucose tolerance test"
)

blood_pressure = st.sidebar.number_input(
    "Blood Pressure (mm Hg)", 
    min_value=40.0, max_value=150.0, value=70.0, step=1.0,
    help="Diastolic blood pressure"
)

skin_thickness = st.sidebar.number_input(
    "Skin Thickness (mm)", 
    min_value=5.0, max_value=50.0, value=20.0, step=1.0,
    help="Triceps skin fold thickness"
)

insulin = st.sidebar.number_input(
    "Insulin Level (μU/ml)", 
    min_value=10.0, max_value=500.0, value=80.0, step=1.0,
    help="2-hour serum insulin"
)

bmi = st.sidebar.number_input(
    "BMI (kg/m²)", 
    min_value=15.0, max_value=50.0, value=25.0, step=0.1,
    help="Body Mass Index"
)

pedigree = st.sidebar.number_input(
    "Diabetes Pedigree Function", 
    min_value=0.0, max_value=2.0, value=0.5, step=0.01,
    help="Genetic predisposition to diabetes"
)

age = st.sidebar.number_input(
    "Age (years)", 
    min_value=18, max_value=100, value=30,
    help="Age in years"
)

# Preprocessing function (based on your notebook)
def preprocess_input(pregnancies, glucose, blood_pressure, skin_thickness, 
                    insulin, bmi, pedigree, age):
    """
    Preprocess input data to match the training pipeline
    """
    # Create input dataframe
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [pedigree],
        'Age': [age]
    })
    
    return input_data

# Rule-based prediction fallback
def rule_based_prediction(pregnancies, glucose, blood_pressure, skin_thickness, 
                         insulin, bmi, pedigree, age):
    """
    Simple rule-based diabetes risk assessment based on medical thresholds
    """
    risk_score = 0
    risk_factors = []
    
    # Glucose risk (highest weight)
    if glucose >= 200:
        risk_score += 40
        risk_factors.append("Diabetic glucose level")
    elif glucose >= 140:
        risk_score += 25
        risk_factors.append("Prediabetic glucose level")
    elif glucose >= 126:
        risk_score += 15
        risk_factors.append("Elevated fasting glucose")
    
    # BMI risk
    if bmi >= 35:
        risk_score += 20
        risk_factors.append("Severe obesity")
    elif bmi >= 30:
        risk_score += 15
        risk_factors.append("Obesity")
    elif bmi >= 25:
        risk_score += 8
        risk_factors.append("Overweight")
    
    # Age risk
    if age >= 65:
        risk_score += 15
        risk_factors.append("Advanced age (≥65)")
    elif age >= 45:
        risk_score += 10
        risk_factors.append("Middle age (≥45)")
    elif age >= 35:
        risk_score += 5
        risk_factors.append("Age ≥35")
    
    # Blood pressure risk
    if blood_pressure >= 90:
        risk_score += 10
        risk_factors.append("High blood pressure")
    elif blood_pressure >= 80:
        risk_score += 5
        risk_factors.append("Elevated blood pressure")
    
    # Pregnancy history
    if pregnancies >= 7:
        risk_score += 10
        risk_factors.append("High parity (≥7 pregnancies)")
    elif pregnancies >= 4:
        risk_score += 5
        risk_factors.append("Multi-parity (≥4 pregnancies)")
    
    # Family history (pedigree function)
    if pedigree >= 1.0:
        risk_score += 10
        risk_factors.append("Strong family history")
    elif pedigree >= 0.5:
        risk_score += 5
        risk_factors.append("Moderate family history")
    
    # Insulin resistance indicators
    if insulin >= 300:
        risk_score += 8
        risk_factors.append("Very high insulin")
    elif insulin >= 200:
        risk_score += 5
        risk_factors.append("Elevated insulin")
    
    # Convert to probability (0-1 scale)
    probability = min(risk_score / 100, 0.95)  # Cap at 95%
    
    # Determine prediction
    prediction = 1 if probability >= 0.5 else 0
    
    return prediction, probability, risk_factors

# Medical interpretation function
def get_medical_interpretation(glucose, bmi, blood_pressure, age, pregnancies):
    """
    Provide medical interpretations for the input values
    """
    interpretations = {
        'glucose_status': 'Normal' if glucose < 140 else 'Prediabetic' if glucose < 200 else 'Diabetic Range',
        'bmi_status': 'Underweight' if bmi < 18.5 else 'Normal' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obese',
        'bp_status': 'Normal' if blood_pressure < 80 else 'Stage 1 HTN' if blood_pressure < 90 else 'Stage 2 HTN',
        'age_group': 'Young Adult' if age < 30 else 'Middle-aged' if age < 50 else 'Senior',
        'pregnancy_risk': 'No pregnancies' if pregnancies == 0 else 'Low parity' if pregnancies <= 2 else 'Multi-parity' if pregnancies <= 5 else 'Grand multi-parity'
    }
    
    return interpretations

# Prediction button
if st.sidebar.button("🔬 Analyze Risk", type="primary"):
    # Preprocess input
    input_data = preprocess_input(pregnancies, glucose, blood_pressure, 
                                skin_thickness, insulin, bmi, pedigree, age)
    
    # Get medical interpretation
    medical_info = get_medical_interpretation(glucose, bmi, blood_pressure, age, pregnancies)
    
    # Make prediction using AI model or rule-based fallback
    try:
        if model is not None:
            # Use AI model
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            prediction_method = "AI Model"
            rule_risk_factors = []
        else:
            # Use rule-based fallback
            prediction, probability, rule_risk_factors = rule_based_prediction(
                pregnancies, glucose, blood_pressure, skin_thickness, 
                insulin, bmi, pedigree, age
            )
            prediction_method = "Rule-Based Assessment"
        
        # Determine risk level
        if probability > 0.7:
            risk_level = "High"
            risk_class = "risk-high"
            risk_icon = "🔴"
        elif probability > 0.3:
            risk_level = "Moderate"
            risk_class = "risk-moderate"
            risk_icon = "🟡"
        else:
            risk_level = "Low"
            risk_class = "risk-low"
            risk_icon = "🟢"
        
        # Main results section
        st.markdown('<h2 class="sub-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
        
        # Show prediction method
        if model is None:
            st.info(f"🔄 **Assessment Method**: {prediction_method} (AI model unavailable)")
        else:
            st.success(f"🤖 **Assessment Method**: {prediction_method}")
        
        # Create columns for results
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="{risk_class}">
                <h3>{risk_icon} Risk Level: {risk_level}</h3>
                <p><strong>Prediction:</strong> {'Diabetes Risk' if prediction == 1 else 'No Diabetes Risk'}</p>
                <p><strong>Confidence:</strong> {probability:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Risk probability display
            st.markdown("### 📊 Risk Probability")
            
            # Create a simple progress bar for risk visualization
            risk_percentage = probability * 100
            st.metric(
                label="Diabetes Risk",
                value=f"{risk_percentage:.1f}%",
                delta=f"{'High' if risk_percentage > 70 else 'Moderate' if risk_percentage > 30 else 'Low'} Risk"
            )
            
            # Simple progress bar
            st.progress(probability)
            
            # Color-coded risk display
            if risk_percentage > 70:
                st.error(f"🔴 High Risk: {risk_percentage:.1f}%")
            elif risk_percentage > 30:
                st.warning(f"🟡 Moderate Risk: {risk_percentage:.1f}%")
            else:
                st.success(f"🟢 Low Risk: {risk_percentage:.1f}%")
        
        with col3:
            st.markdown("### 📋 Medical Assessment")
            st.write(f"**Glucose Status:** {medical_info['glucose_status']}")
            st.write(f"**BMI Category:** {medical_info['bmi_status']}")
            st.write(f"**Blood Pressure:** {medical_info['bp_status']}")
            st.write(f"**Age Group:** {medical_info['age_group']}")
            st.write(f"**Pregnancy History:** {medical_info['pregnancy_risk']}")
        
        # Risk factors section
        st.markdown('<h3 class="sub-header">⚠️ Risk Factors Analysis</h3>', unsafe_allow_html=True)
        
        # Combine rule-based and medical risk factors
        all_risk_factors = rule_risk_factors if rule_risk_factors else []
        
        if all_risk_factors:
            st.warning("**Identified Risk Factors:**")
            for factor in all_risk_factors:
                st.write(f"• {factor}")
        else:
            st.success("✅ No major risk factors identified")
        
        # Feature values visualization
        st.markdown('<h3 class="sub-header">📈 Input Values Summary</h3>', unsafe_allow_html=True)
        
        # Create a simple table for patient profile
        profile_data = {
            'Parameter': ['Pregnancies', 'Glucose (mg/dL)', 'Blood Pressure (mm Hg)', 
                         'Skin Thickness (mm)', 'Insulin (μU/ml)', 'BMI (kg/m²)', 
                         'Pedigree Function', 'Age (years)'],
            'Value': [pregnancies, glucose, blood_pressure, skin_thickness, 
                     insulin, bmi, pedigree, age],
            'Status': [
                f"{pregnancies} pregnancies",
                medical_info['glucose_status'],
                medical_info['bp_status'],
                'Normal' if skin_thickness <= 18 else 'Above normal',
                f"{insulin} μU/ml",
                medical_info['bmi_status'],
                f"{pedigree:.3f}",
                medical_info['age_group']
            ]
        }
        
        profile_df = pd.DataFrame(profile_data)
        st.dataframe(profile_df, use_container_width=True)
        
        # Simple bar chart using streamlit
        st.markdown("### 📊 Parameter Values (Normalized)")
        chart_data = pd.DataFrame({
            'Parameters': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                          'Insulin', 'BMI', 'Pedigree', 'Age'],
            'Normalized Values': [
                pregnancies/10, glucose/200, blood_pressure/100, skin_thickness/40,
                insulin/300, bmi/40, pedigree*50, age/80
            ]
        })
        st.bar_chart(chart_data.set_index('Parameters'))
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please try refreshing the page or contact support if the issue persists.")

# Information section
st.markdown('<h2 class="sub-header">ℹ️ About This System</h2>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    ### 🎯 Model Information
    - **Algorithm**: Advanced Machine Learning Ensemble
    - **Training Data**: Pima Indian Diabetes Database
    - **Features**: 8 clinical measurements
    - **Accuracy**: High-performance validated model
    - **Purpose**: Early diabetes risk assessment
    """)

with col2:
    st.markdown("""
    ### ⚕️ Medical Disclaimer
    - This tool is for **screening purposes only**
    - **Not a substitute** for professional medical advice
    - Results should be **discussed with healthcare providers**
    - For **emergency symptoms**, seek immediate medical care
    - Regular health checkups are recommended
    """)

# Sample test cases
st.markdown('<h3 class="sub-header">🧪 Try Sample Cases</h3>', unsafe_allow_html=True)

sample_cases = {
    "High Risk Patient": {
        "pregnancies": 8, "glucose": 183, "blood_pressure": 95, 
        "skin_thickness": 35, "insulin": 250, "bmi": 35.2, 
        "pedigree": 0.672, "age": 55
    },
    "Low Risk Patient": {
        "pregnancies": 1, "glucose": 95, "blood_pressure": 65, 
        "skin_thickness": 20, "insulin": 85, "bmi": 23.1, 
        "pedigree": 0.167, "age": 25
    },
    "Moderate Risk Patient": {
        "pregnancies": 3, "glucose": 125, "blood_pressure": 82, 
        "skin_thickness": 28, "insulin": 150, "bmi": 28.5, 
        "pedigree": 0.45, "age": 38
    }
}

selected_case = st.selectbox("Select a sample case to test:", 
                           ["Choose a sample case..."] + list(sample_cases.keys()))

if selected_case != "Choose a sample case...":
    case_data = sample_cases[selected_case]
    st.json(case_data)
    st.info("💡 Copy these values to the sidebar inputs to test this case!")

# Footer
st.markdown("""
---
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🏥 Diabetes Risk Prediction System | Powered by AI & Medical Knowledge</p>
    <p><em>Developed for educational and screening purposes</em></p>
</div>
""", unsafe_allow_html=True)