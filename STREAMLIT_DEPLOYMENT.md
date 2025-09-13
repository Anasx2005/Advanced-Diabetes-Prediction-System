# Diabetes Prediction Streamlit App Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- The trained model file `diabetes_model.pkl` (generated from your Jupyter notebook)

### Installation Steps

1. **Navigate to the project directory:**
   ```bash
   cd "c:\Users\LENOVO\OneDrive - Faculty of Science, Helwan University (1)\Desktop\GTC Internship\Project_02"
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

3. **Ensure model file exists:**
   - Make sure `diabetes_model.pkl` is in the same directory as `app.py`
   - If not, run your Jupyter notebook to generate the model file

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

5. **Access the app:**
   - The app will automatically open in your default web browser
   - Default URL: `http://localhost:8501`

## 📋 Features

### 🩺 **Medical Input Interface**
- Comprehensive patient information form
- Input validation and help text
- Medical reference ranges provided

### 🎯 **Advanced Prediction System**
- Real-time diabetes risk prediction
- Probability scoring with confidence levels
- Risk stratification (Low/Moderate/High)

### 📊 **Visual Analytics**
- Interactive risk gauge meter
- Color-coded medical assessments
- Professional medical interpretations

### 💡 **Clinical Recommendations**
- Risk-specific recommendations
- Evidence-based medical guidelines
- Actionable next steps for healthcare providers

## 🔧 Technical Architecture

### **Preprocessing Pipeline**
- Target-aware missing value imputation
- Medical knowledge-based feature engineering
- Robust scaling for outlier resistance
- One-hot encoding for categorical features

### **Model Integration**
- Seamless integration with trained ML model
- Consistent preprocessing between training and prediction
- Error handling and validation

### **User Interface**
- Responsive design with Plotly visualizations
- Professional medical styling
- Intuitive navigation and clear information hierarchy

## 📱 Deployment Options

### **Local Development**
```bash
streamlit run app.py
```

### **Streamlit Cloud (Recommended)**
1. Push your code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy directly from repository
4. Automatic updates on code changes

### **Docker Deployment**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### **Heroku Deployment**
1. Create `Procfile`:
   ```
   web: sh setup.sh && streamlit run app.py
   ```

2. Create `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]
   port = $PORT
   enableCORS = false
   headless = true
   " > ~/.streamlit/config.toml
   ```

## 🔍 Troubleshooting

### **Common Issues**

1. **Model file not found:**
   - Ensure `diabetes_model.pkl` exists in the app directory
   - Re-run the Jupyter notebook to generate the model

2. **Package conflicts:**
   - Use virtual environment
   - Update package versions in requirements_streamlit.txt

3. **Prediction errors:**
   - Check input data formats
   - Verify model compatibility

### **Performance Optimization**
- Model caching with `@st.cache_resource`
- Efficient preprocessing pipeline
- Minimal dependencies for faster loading

## 📋 Usage Instructions

### **For Healthcare Providers:**
1. Enter patient diagnostic measurements
2. Review prediction results and confidence levels
3. Consider medical interpretations and risk factors
4. Follow clinical recommendations based on risk level

### **Input Parameters:**
- **Pregnancies:** Number of pregnancies (0-17)
- **Glucose:** Plasma glucose concentration (mg/dL)
- **Blood Pressure:** Diastolic blood pressure (mm Hg)
- **Skin Thickness:** Triceps skin fold thickness (mm)
- **Insulin:** 2-hour serum insulin (μU/ml)
- **BMI:** Body mass index (kg/m²)
- **Pedigree Function:** Diabetes genetic predisposition score
- **Age:** Patient age in years

## 🏥 Medical Disclaimers

⚠️ **Important Medical Notice:**
- This tool is for educational and research purposes
- Not intended for clinical diagnosis
- Always consult qualified healthcare professionals
- Results should be interpreted by medical experts
- Regular medical checkups recommended regardless of predictions

## 📊 Model Performance

- **Training Accuracy:** >85% (varies by best model selected)
- **Cross-validation:** 10-fold validation implemented
- **Features:** Advanced medical knowledge integration
- **Validation:** Comprehensive test set evaluation

## 🔄 Updates and Maintenance

### **Model Updates:**
- Retrain model with new data periodically
- Replace `diabetes_model.pkl` with updated version
- Test thoroughly before deployment

### **App Updates:**
- Monitor Streamlit updates for compatibility
- Update requirements.txt as needed
- Test all features after updates

## 📞 Support

For technical issues or questions:
- Check troubleshooting section
- Review error messages carefully
- Ensure all dependencies are correctly installed
- Verify model file integrity

---

**Created by:** GTC Internship Project Team  
**Last Updated:** September 2025  
**Version:** 1.0.0