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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .risk-high {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .risk-moderate {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained diabetes prediction model"""
    try:
        model = joblib.load('diabetes_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'diabetes_model.pkl' not found. Please ensure the model file is in the same directory as this app.")
        return None

def predict_diabetes_risk(pregnancies, glucose, blood_pressure, skin_thickness, 
                         insulin, bmi, pedigree, age, model):
    """
    Predict diabetes risk with comprehensive feature engineering
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
    
    # Apply the same preprocessing pipeline as in training
    processed_data = input_data.copy()
    
    # 1. Create categorical features
    processed_data['Glucose_Category'] = pd.cut(processed_data['Glucose'], 
                                               bins=[0, 139, 200, float('inf')], 
                                               labels=["Normal", "Prediabetes", "Diabetes"])
    
    processed_data['BMI_Category'] = pd.cut(processed_data['BMI'], 
                                           bins=[0, 18.5, 24.9, 29.9, float('inf')], 
                                           labels=["Underweight", "Normal", "Overweight", "Obese"])
    
    processed_data['BP_Category'] = pd.cut(processed_data['BloodPressure'], 
                                          bins=[0, 79, 89, float('inf')], 
                                          labels=["Normal", "Stage1_HTN", "Stage2_HTN"])
    
    processed_data['SkinThickness_Normal'] = (processed_data['SkinThickness'] <= 18.0).astype(int)
    
    processed_data['Age_Group'] = pd.cut(processed_data['Age'], 
                                        bins=[0, 30, 40, 50, float('inf')], 
                                        labels=["Young", "Middle_Young", "Middle_Aged", "Senior"])
    
    processed_data['Pregnancy_Risk'] = pd.cut(processed_data['Pregnancies'], 
                                             bins=[-1, 0, 2, 5, float('inf')], 
                                             labels=["Nulliparous", "Low_Parity", "Multi_Parity", "Grand_Multi"])
    
    # 2. One-hot encoding
    categorical_features_pred = ['Glucose_Category', 'BMI_Category', 'BP_Category', 
                                'Age_Group', 'Pregnancy_Risk']
    processed_data_encoded = pd.get_dummies(processed_data, columns=categorical_features_pred, 
                                           drop_first=True, dtype=int)
    
    # 3. Define expected feature columns (from training)
    expected_features = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age', 'SkinThickness_Normal',
        'Glucose_Category_Diabetes', 'Glucose_Category_Prediabetes',
        'BMI_Category_Normal', 'BMI_Category_Obese', 'BMI_Category_Overweight',
        'BP_Category_Normal', 'BP_Category_Stage2_HTN',
        'Age_Group_Middle_Aged', 'Age_Group_Middle_Young', 'Age_Group_Senior',
        'Pregnancy_Risk_Grand_Multi', 'Pregnancy_Risk_Low_Parity', 'Pregnancy_Risk_Multi_Parity'
    ]
    
    # 4. Ensure all required columns are present
    for col in expected_features:
        if col not in processed_data_encoded.columns:
            processed_data_encoded[col] = 0
    
    # 5. Apply robust scaling (simplified for demo - using approximate training stats)
    numerical_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    # Approximate population statistics for scaling
    scaling_params = {
        'Pregnancies': {'median': 3.0, 'iqr': 4.0},
        'Glucose': {'median': 117.0, 'iqr': 42.0},
        'BloodPressure': {'median': 72.0, 'iqr': 22.0},
        'SkinThickness': {'median': 23.0, 'iqr': 18.0},
        'Insulin': {'median': 30.5, 'iqr': 140.0},
        'BMI': {'median': 32.0, 'iqr': 8.0},
        'DiabetesPedigreeFunction': {'median': 0.372, 'iqr': 0.4},
        'Age': {'median': 29.0, 'iqr': 22.0}
    }
    
    for feature in numerical_features:
        median_val = scaling_params[feature]['median']
        iqr = scaling_params[feature]['iqr']
        processed_data_encoded[feature] = (processed_data_encoded[feature] - median_val) / iqr
    
    # 6. Select and reorder columns to match training data
    prediction_input = processed_data_encoded[expected_features]
    
    # Make prediction
    try:
        prediction = model.predict(prediction_input)[0]
        probability = model.predict_proba(prediction_input)[0][1]
        
        return {
            'prediction': 'Diabetes' if prediction == 1 else 'No Diabetes',
            'probability': probability,
            'risk_level': 'High' if probability > 0.7 else 'Moderate' if probability > 0.3 else 'Low',
            'confidence': 'High' if abs(probability - 0.5) > 0.3 else 'Moderate' if abs(probability - 0.5) > 0.15 else 'Low'
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def get_medical_interpretations(glucose, bmi, blood_pressure, age, pregnancies):
    """Generate medical interpretations for the input values"""
    interpretations = {}
    
    # Glucose interpretation
    if glucose < 140:
        interpretations['glucose'] = ('Normal', 'green')
    elif glucose < 200:
        interpretations['glucose'] = ('Prediabetes', 'orange')
    else:
        interpretations['glucose'] = ('Diabetes Range', 'red')
    
    # BMI interpretation
    if bmi < 18.5:
        interpretations['bmi'] = ('Underweight', 'blue')
    elif bmi < 25:
        interpretations['bmi'] = ('Normal', 'green')
    elif bmi < 30:
        interpretations['bmi'] = ('Overweight', 'orange')
    else:
        interpretations['bmi'] = ('Obese', 'red')
    
    # Blood pressure interpretation
    if blood_pressure < 80:
        interpretations['bp'] = ('Normal', 'green')
    elif blood_pressure < 90:
        interpretations['bp'] = ('Stage 1 HTN', 'orange')
    else:
        interpretations['bp'] = ('Stage 2 HTN', 'red')
    
    # Age group
    if age < 30:
        interpretations['age'] = ('Young Adult', 'green')
    elif age < 50:
        interpretations['age'] = ('Middle-aged', 'orange')
    else:
        interpretations['age'] = ('Senior', 'red')
    
    return interpretations

def create_risk_gauge(probability):
    """Create a risk gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Diabetes Risk (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    # Title
    st.markdown('<h1 class="main-header">🩺 Diabetes Risk Prediction System</h1>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Sidebar for input
    st.sidebar.header("Patient Information")
    st.sidebar.markdown("Enter the patient's medical information below:")
    
    # Input fields with validation and help text
    pregnancies = st.sidebar.number_input(
        "Number of Pregnancies", 
        min_value=0, max_value=17, value=1,
        help="Number of times pregnant (0-17)"
    )
    
    glucose = st.sidebar.number_input(
        "Plasma Glucose Concentration (mg/dL)", 
        min_value=0, max_value=300, value=120,
        help="2-hour oral glucose tolerance test result"
    )
    
    blood_pressure = st.sidebar.number_input(
        "Diastolic Blood Pressure (mm Hg)", 
        min_value=0, max_value=150, value=70,
        help="Diastolic blood pressure measurement"
    )
    
    skin_thickness = st.sidebar.number_input(
        "Triceps Skin Fold Thickness (mm)", 
        min_value=0, max_value=100, value=20,
        help="Triceps skin fold thickness measurement"
    )
    
    insulin = st.sidebar.number_input(
        "2-Hour Serum Insulin (μU/ml)", 
        min_value=0, max_value=1000, value=80,
        help="2-hour serum insulin level"
    )
    
    bmi = st.sidebar.number_input(
        "Body Mass Index (kg/m²)", 
        min_value=10.0, max_value=50.0, value=25.0, step=0.1,
        help="Body Mass Index calculation"
    )
    
    pedigree = st.sidebar.number_input(
        "Diabetes Pedigree Function", 
        min_value=0.0, max_value=3.0, value=0.5, step=0.001,
        help="Genetic predisposition score"
    )
    
    age = st.sidebar.number_input(
        "Age (years)", 
        min_value=21, max_value=100, value=30,
        help="Patient age in years"
    )
    
    # Prediction button
    if st.sidebar.button("🔍 Predict Diabetes Risk", type="primary"):
        # Make prediction
        result = predict_diabetes_risk(pregnancies, glucose, blood_pressure, skin_thickness,
                                     insulin, bmi, pedigree, age, model)
        
        if result:
            # Main results section
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                # Risk gauge
                fig = create_risk_gauge(result['probability'])
                st.plotly_chart(fig, use_container_width=True)
            
            # Results display
            st.markdown("---")
            
            # Risk level display with color coding
            risk_color = {
                'High': 'red',
                'Moderate': 'orange', 
                'Low': 'green'
            }
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Prediction", result['prediction'])
            
            with col2:
                st.metric("Risk Probability", f"{result['probability']:.1%}")
            
            with col3:
                st.markdown(f"**Risk Level:** :{risk_color[result['risk_level']]}[{result['risk_level']}]")
            
            with col4:
                st.markdown(f"**Confidence:** {result['confidence']}")
            
            # Medical interpretations
            st.markdown("---")
            st.subheader("📋 Medical Assessment")
            
            interpretations = get_medical_interpretations(glucose, bmi, blood_pressure, age, pregnancies)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Clinical Parameters:**")
                for param, (status, color) in interpretations.items():
                    param_names = {
                        'glucose': 'Glucose Status',
                        'bmi': 'BMI Category', 
                        'bp': 'Blood Pressure',
                        'age': 'Age Group'
                    }
                    st.markdown(f"• {param_names[param]}: :{color}[{status}]")
            
            with col2:
                st.markdown("**Risk Factors Identified:**")
                risk_factors = []
                
                if glucose >= 140:
                    risk_factors.append("🔸 Elevated glucose levels")
                if bmi >= 30:
                    risk_factors.append("🔸 Obesity")
                if blood_pressure >= 90:
                    risk_factors.append("🔸 Hypertension")
                if age >= 45:
                    risk_factors.append("🔸 Advanced age")
                if pregnancies >= 5:
                    risk_factors.append("🔸 High parity")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(factor)
                else:
                    st.markdown("🟢 No major risk factors identified")
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Clinical Recommendations")
            
            if result['risk_level'] == 'High':
                st.error("""
                **High Risk Patient - Immediate Action Required:**
                - Schedule comprehensive diabetes evaluation
                - Consider HbA1c testing
                - Implement intensive lifestyle interventions
                - Monitor blood glucose regularly
                - Consult endocrinologist if needed
                """)
            elif result['risk_level'] == 'Moderate':
                st.warning("""
                **Moderate Risk Patient - Preventive Measures:**
                - Annual diabetes screening recommended
                - Lifestyle modification counseling
                - Weight management program
                - Regular exercise routine
                - Dietary consultation
                """)
            else:
                st.success("""
                **Low Risk Patient - Maintenance:**
                - Continue healthy lifestyle
                - Regular health checkups
                - Maintain healthy weight
                - Stay physically active
                - Monitor if risk factors develop
                """)
    
    # Information section
    st.markdown("---")
    st.subheader("ℹ️ About This System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Information:**
        - Advanced machine learning model trained on diabetes dataset
        - Features comprehensive medical knowledge integration
        - Includes robust preprocessing and feature engineering
        - Provides risk stratification with medical context
        """)
    
    with col2:
        st.markdown("""
        **Medical Reference Ranges:**
        - **Glucose:** Normal <140, Prediabetes 140-200, Diabetes >200 mg/dL
        - **BMI:** Normal 18.5-24.9, Overweight 25-29.9, Obese ≥30 kg/m²
        - **Blood Pressure:** Normal <80, Stage 1 HTN 80-89, Stage 2 HTN ≥90 mmHg
        - **Age Groups:** Young <30, Middle-aged 30-50, Senior >50 years
        """)

if __name__ == "__main__":
    main()