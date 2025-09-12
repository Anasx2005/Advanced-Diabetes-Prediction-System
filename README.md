# Advanced Diabetes Prediction System

## Overview
This project develops a robust, clinically informed machine learning system to predict diabetes status (binary outcome) using the Pima Indians Diabetes dataset. The workflow integrates medical domain knowledge, rigorous preprocessing, feature engineering, model benchmarking.

## Deployment 
you can see my progect entire this link " https://advanced-diabetes-prediction-system-faoaukgccytpmytbyusj3b.streamlit.app/ "

## Objectives
- Clean and preprocess clinically noisy tabular health data.
- Engineer medically meaningful categorical and composite features.
- Evaluate a broad portfolio of algorithms (linear, tree-based, boosting, kernel, distance-based).
- Optimize top-performing ensemble models via systematic hyperparameter tuning.
- Deliver a reproducible, extensible pipeline and clinically interpretable prediction function.

## Dataset
- Samples: 768 female patients 
- Original Features (8 predictors + target):
  - Pregnancies
  - Glucose (2-hr plasma glucose concentration in OGTT)
  - BloodPressure (Diastolic, mm Hg)
  - SkinThickness (Triceps skinfold thickness, mm)
  - Insulin (2-hour serum insulin, μU/ml)
  - BMI (kg/m²)
  - DiabetesPedigreeFunction (genetic predisposition index)
  - Age (years)
  - Outcome (0 = No Diabetes, 1 = Diabetes)

## Clinical Reference Ranges Incorporated
- Glucose (OGTT): <140 normal, 140–199 prediabetes, ≥200 diagnostic threshold (ADA).
- Blood Pressure (diastolic bands used here): <80 normal, 80–89 Stage 1, ≥90 Stage 2 (ACC/AHA derived categories).
- BMI: <18.5 underweight; 18.5–24.9 normal; 25.0–29.9 overweight; ≥30 obese (WHO).
- SkinThickness: ≤18 mm flagged as “normal” surrogate threshold (contextual heuristic for binary simplification).

## End-to-End Pipeline
1. Data Ingestion
   - Load CSV; inspect schema, class balance, and numeric distributions.
2. Integrity & Missingness Handling
   - Replace medically implausible zero values in: Glucose, BloodPressure, SkinThickness, Insulin, BMI with NaN.
   - Apply target-aware median imputation (class-conditional medians) to preserve outcome-specific central tendencies.
3. Outlier Management
   - Identify outliers per feature using modified IQR on 10th–90th percentile window.
   - Cap (winsorize) values beyond computed low/high thresholds (retain distribution robustness without deletion).
4. Feature Engineering (Medical Knowledge)
   - Glucose_Category: Normal / Prediabetes / Diabetes (binned).
   - BMI_Category: Underweight / Normal / Overweight / Obese.
   - BP_Category: Normal / Stage1_HTN / Stage2_HTN (diastolic bands).
   - SkinThickness_Normal: Binary flag (≤18 mm).
   - Age_Group: Young / Middle_Young / Middle_Aged / Senior (progressive decades).
   - Pregnancy_Risk: Nulliparous / Low_Parity / Multi_Parity / Grand_Multi.
5. Encoding
   - One-hot encode all engineered categorical features (drop-first to reduce collinearity).
6. Scaling
   - Apply robust scaling (median and IQR) to continuous numerical variables (resistant to remaining outliers). Custom implementation to mirror logic in single-sample inference.
7. Modeling & Benchmarking
   - Baseline algorithms: Logistic Regression, KNN, Decision Tree, Random Forest, SVM, XGBoost, Gradient Boosting, LightGBM.
   - 10-fold stratified cross-validation for accuracy estimation.
   - Class imbalance addressed via class_weight='balanced' where applicable (no synthetic oversampling introduced for transparency).
8. Hyperparameter Optimization (Top Models)
   - Grid search (5-fold CV) on top-performing ensemble models (e.g., Gradient Boosting, LightGBM, Random Forest, XGBoost).
   - Parameter grids include depth, learning rate, estimators, subsampling, regularization dimensions.
9. Final Model Selection
   - Select best tuned model based on held-out test accuracy (with ROC AUC where available).
10. Evaluation & Reporting
   - Confusion matrix heatmap.
   - ROC curve and AUC.
   - Classification report (precision, recall, F1, support).
11. Prediction Interface
   - Function: `predict_diabetes_risk_enhanced()`
   - Accepts raw clinical-style measurements.
   - Re-applies deterministic feature engineering and scaling pipeline.
   - Produces: predicted class, probability, risk tier (Low / Moderate / High), confidence, medical context summary, risk factor list.

## Key Design Decisions
| Aspect | Rationale |
|--------|-----------|
| Target-aware imputation | Preserves class-dependent central tendencies vs. global mean/median. |
| Outlier capping (not removal) | Maintains sample size; avoids artificial class skew. |
| Robust scaling | Mitigates residual outlier influence on distance/kernel models. |
| Categorical medical bins | Injects domain priors; improves interpretability & potential separability. |
| Ensemble + linear mix | Balances variance reduction with explainability. |
| No synthetic sampling | Prefer intrinsic class_weight adjustments to avoid potential signal distortion. |

## Reproducibility
- Deterministic splits via fixed random_state (42).
- All transformation logic implemented explicitly (no hidden stateful pipelines omitted).
- Scaling and encoding logic replicated for inference path to ensure feature alignment.

## Usage (Notebook-Driven)
1. Open the notebook `Enhanced_Diabetes_Prediction_Professional.ipynb`.
2. Execute cells sequentially through model selection section.
3. (Optional) Inspect intermediate dataframes for validation.
4. Use the test cases or call `predict_diabetes_risk_enhanced()` directly, e.g.:
```python
predict_diabetes_risk_enhanced(
    pregnancies=3,
    glucose=145,
    blood_pressure=82,
    skin_thickness=28,
    insulin=130,
    bmi=31.5,
    pedigree=0.45,
    age=44
)
```


