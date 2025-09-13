import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        return model
    except FileNotFoundError:
        st.error("Model file 'diabetes_model.pkl' not found. Please ensure the model file is in the same directory.")
        return None

model = load_model()

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
    
    # Risk factors
    risk_factors = []
    if glucose >= 140: risk_factors.append('Elevated glucose')
    if bmi >= 30: risk_factors.append('Obesity')
    if blood_pressure >= 90: risk_factors.append('Hypertension')
    if age >= 45: risk_factors.append('Advanced age')
    if pregnancies >= 5: risk_factors.append('High parity')
    
    interpretations['risk_factors'] = risk_factors if risk_factors else ['No major risk factors identified']
    
    return interpretations

# Prediction button
if st.sidebar.button("🔬 Analyze Risk", type="primary"):
    if model is not None:
        # Preprocess input
        input_data = preprocess_input(pregnancies, glucose, blood_pressure, 
                                    skin_thickness, insulin, bmi, pedigree, age)
        
        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            # Get medical interpretation
            medical_info = get_medical_interpretation(glucose, bmi, blood_pressure, age, pregnancies)
            
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
                # Risk probability gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Diabetes Risk (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col3:
                st.markdown("### 📋 Medical Assessment")
                st.write(f"**Glucose Status:** {medical_info['glucose_status']}")
                st.write(f"**BMI Category:** {medical_info['bmi_status']}")
                st.write(f"**Blood Pressure:** {medical_info['bp_status']}")
                st.write(f"**Age Group:** {medical_info['age_group']}")
                st.write(f"**Pregnancy History:** {medical_info['pregnancy_risk']}")
            
            # Risk factors section
            st.markdown('<h3 class="sub-header">⚠️ Risk Factors Analysis</h3>', unsafe_allow_html=True)
            
            if len(medical_info['risk_factors']) > 1 or medical_info['risk_factors'][0] != 'No major risk factors identified':
                st.warning("**Identified Risk Factors:**")
                for factor in medical_info['risk_factors']:
                    st.write(f"• {factor}")
            else:
                st.success("✅ No major risk factors identified")
            
            # Feature importance visualization
            st.markdown('<h3 class="sub-header">📈 Input Values Visualization</h3>', unsafe_allow_html=True)
            
            # Create radar chart for patient profile
            categories = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                         'Insulin', 'BMI', 'Pedigree Function', 'Age']
            
            values = [pregnancies/10, glucose/200, blood_pressure/100, skin_thickness/40,
                     insulin/300, bmi/40, pedigree*50, age/80]
            
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Patient Profile'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Patient Profile Radar Chart"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    else:
        st.error("Model not loaded. Cannot make predictions.")

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