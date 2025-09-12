import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Load the trained model and preprocessing components
@st.cache_resource
def load_model_and_preprocessors():
    """Load the trained model and preprocessing components"""
    try:
        # Load the saved model
        model = joblib.load("diabetes_model.pkl")
        
        # Load the dataset to recreate preprocessing components
        data = pd.read_csv('diabetes.csv')
        
        # Recreate preprocessing pipeline
        zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        data_clean = data.copy()
        data_clean[zero_columns] = data_clean[zero_columns].replace(0, np.nan)
        
        X = data_clean.drop('Outcome', axis=1)
        y = data_clean['Outcome']
        X = X.drop('Insulin', axis=1)  # Remove Insulin column
        
        # Recreate imputer and scaler
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
        
        scaler = StandardScaler()
        scaler.fit(X_imputed)
        
        return model, imputer, scaler, X.columns.tolist()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

def predict_diabetes_risk(pregnancies, glucose, blood_pressure, skin_thickness, 
                         bmi, pedigree, age, model, imputer, scaler, feature_columns):
    """
    Predict diabetes risk for a new patient using the loaded model
    """
    # Prepare input data (without Insulin)
    patient_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                             bmi, pedigree, age]])
    patient_df = pd.DataFrame(patient_data, columns=feature_columns)
    
    # Apply preprocessing pipeline
    patient_imputed = imputer.transform(patient_df)
    patient_scaled = scaler.transform(patient_imputed)
    
    # Make prediction
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0][1]
    
    # Format results
    result = {
        'prediction': 'Diabetes' if prediction == 1 else 'No Diabetes',
        'probability': probability,
        'risk_level': 'High' if probability > 0.7 else 'Moderate' if probability > 0.3 else 'Low'
    }
    
    return result

def get_risk_interpretation(glucose, bmi, blood_pressure, age, pregnancies):
    """Provide medical interpretation of risk factors"""
    risk_factors = []
    
    if glucose >= 126:
        risk_factors.append("Elevated fasting glucose (diabetes range)")
    elif glucose >= 100:
        risk_factors.append("Elevated fasting glucose (prediabetes range)")
    
    if bmi >= 30:
        risk_factors.append("Obesity (BMI ≥ 30)")
    elif bmi >= 25:
        risk_factors.append("Overweight (BMI 25-29.9)")
    
    if blood_pressure >= 90:
        risk_factors.append("High blood pressure (Stage 2)")
    elif blood_pressure >= 80:
        risk_factors.append("Elevated blood pressure (Stage 1)")
    
    if age >= 45:
        risk_factors.append("Advanced age (≥45 years)")
    
    if pregnancies >= 4:
        risk_factors.append("High parity (multiple pregnancies)")
    
    return risk_factors if risk_factors else ["No major risk factors identified"]

# Main application
def main():
    # Load model and preprocessors
    model, imputer, scaler, feature_columns = load_model_and_preprocessors()
    
    if model is None:
        st.error("Failed to load the diabetes prediction model. Please ensure 'diabetes_model.pkl' and 'diabetes.csv' are available.")
        return
    
    # Title and description
    st.title("🏥 Diabetes Risk Prediction System")
    st.markdown("""
    This application uses machine learning to assess diabetes risk based on diagnostic measurements.
    Please enter your health information below to get a personalized risk assessment.
    
    **Disclaimer**: This tool is for educational purposes only and should not replace professional medical advice.
    """)
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Patient Information")
        
        # Input fields
        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0, max_value=20, value=1,
            help="Total number of pregnancies"
        )
        
        glucose = st.number_input(
            "Glucose Level (mg/dL)",
            min_value=50, max_value=300, value=120,
            help="Plasma glucose concentration (normal: <100 mg/dL fasting)"
        )
        
        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)",
            min_value=40, max_value=200, value=80,
            help="Diastolic blood pressure (normal: <80 mm Hg)"
        )
        
        skin_thickness = st.number_input(
            "Skin Thickness (mm)",
            min_value=5, max_value=100, value=20,
            help="Triceps skin fold thickness"
        )
    
    with col2:
        st.subheader("📊 Health Metrics")
        
        bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=10.0, max_value=70.0, value=25.0, step=0.1,
            help="Body Mass Index (normal: 18.5-24.9)"
        )
        
        pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0, max_value=3.0, value=0.5, step=0.01,
            help="Genetic predisposition score"
        )
        
        age = st.number_input(
            "Age (years)",
            min_value=18, max_value=100, value=30,
            help="Age in years"
        )
    
    # Prediction button
    if st.button("🔍 Assess Diabetes Risk", type="primary"):
        # Make prediction
        result = predict_diabetes_risk(
            pregnancies, glucose, blood_pressure, skin_thickness,
            bmi, pedigree, age, model, imputer, scaler, feature_columns
        )
        
        # Display results
        st.subheader("📈 Risk Assessment Results")
        
        # Create columns for results
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.metric("Prediction", result['prediction'])
        
        with res_col2:
            st.metric("Risk Probability", f"{result['probability']:.1%}")
        
        with res_col3:
            risk_color = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}
            st.metric("Risk Level", f"{risk_color.get(result['risk_level'], '')} {result['risk_level']}")
        
        # Risk interpretation
        st.subheader("🩺 Medical Interpretation")
        risk_factors = get_risk_interpretation(glucose, bmi, blood_pressure, age, pregnancies)
        
        if "No major risk factors identified" in risk_factors:
            st.success("✅ No major risk factors identified based on the provided information.")
        else:
            st.warning("⚠️ The following risk factors were identified:")
            for factor in risk_factors:
                st.write(f"• {factor}")
        
        # Visual representation
        st.subheader("📊 Risk Visualization")
        
        # Create a simple gauge chart
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Risk probability bar
        colors = ['green' if result['probability'] < 0.3 else 'orange' if result['probability'] < 0.7 else 'red']
        bars = ax.barh(['Risk Probability'], [result['probability']], color=colors)
        
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability')
        ax.set_title('Diabetes Risk Assessment')
        
        # Add percentage text on bar
        ax.text(result['probability']/2, 0, f"{result['probability']:.1%}", 
                ha='center', va='center', fontweight='bold', color='white')
        
        st.pyplot(fig)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if result['risk_level'] == 'High':
            st.error("""
            **High Risk Detected**
            - Consult with a healthcare provider immediately
            - Consider glucose tolerance testing
            - Discuss lifestyle modifications and potential treatment options
            - Regular monitoring recommended
            """)
        elif result['risk_level'] == 'Moderate':
            st.warning("""
            **Moderate Risk Detected**
            - Schedule a check-up with your healthcare provider
            - Consider lifestyle modifications (diet and exercise)
            - Monitor blood glucose levels regularly
            - Annual diabetes screening recommended
            """)
        else:
            st.success("""
            **Low Risk Assessment**
            - Continue maintaining healthy lifestyle habits
            - Regular exercise and balanced diet
            - Routine health check-ups as recommended
            - Stay informed about diabetes prevention
            """)
    
    # Sidebar with additional information
    st.sidebar.header("ℹ️ About This Tool")
    st.sidebar.markdown("""
    **Model Information:**
    - Trained on Pima Indians Diabetes Database
    - Uses advanced machine learning algorithms
    - Features: 7 diagnostic measurements
    - Accuracy: ~77-80% on test data
    
    **Risk Levels:**
    - **Low**: <30% probability
    - **Moderate**: 30-70% probability  
    - **High**: >70% probability
    
    **Important Notes:**
    - This is a screening tool, not a diagnostic test
    - Always consult healthcare professionals
    - Results based on limited feature set
    - Individual factors may vary
    """)

if __name__ == "__main__":
    main()