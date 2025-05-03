# Thyroid Cancer Risk Predictor

## Overview
This project implements a machine learning model to predict thyroid cancer risk based on patient data. It was developed as a final project for CS 439: Machine Learning Applications in Healthcare.

## Project Description
Thyroid cancer incidence has been increasing globally, and early detection is critical for improved treatment outcomes. This project addresses this challenge by:

* Creating a predictive model for thyroid cancer risk assessment
* Identifying key risk factors that contribute to malignancy
* Providing interpretable results that can aid medical decision-making

The model uses demographic information, clinical measurements, and lifestyle factors to predict the likelihood of thyroid cancer with reasonable accuracy.

## Dataset
The model is trained on a dataset containing patient records with the following features:

* **Demographics**: Age, Gender, Country, Ethnicity
* **Medical History**: Family History, Diabetes
* **Clinical Measurements**: TSH Level, T3 Level, T4 Level, Nodule Size
* **Lifestyle Factors**: Radiation Exposure, Iodine Deficiency, Smoking, Obesity
* **Labels**: Thyroid Cancer Risk (Low/Medium/High), Diagnosis (Benign/Malignant)

## Model Implementation
The project implements two machine learning models:
1. **Random Forest Classifier** - Main predictive model
2. **Logistic Regression** - Baseline comparison model

Both models are evaluated using multiple metrics including accuracy, precision, recall, and F1-score.

## Key Features
* **Data Visualization**: Exploratory data analysis with meaningful visualizations
* **Feature Importance Analysis**: Understanding which factors most strongly predict thyroid cancer
* **Model Comparison**: Performance analysis between different algorithms
* **Patient Prediction**: Functionality to predict cancer risk for new patients

## Results
The models achieve comparable performance:
* Random Forest: 82.3% accuracy
* Logistic Regression: 82.5% accuracy

Key findings include:
* Thyroid Cancer Risk assessment is highly predictive of actual diagnosis
* Demographic factors including Ethnicity and Family History are important predictors
* Clinical measurements show varying levels of importance in the model

## Usage
To use this thyroid cancer predictor:

```python
# Load the trained model
predictor = ThyroidCancerPredictor()

# Example patient data
new_patient_data = pd.DataFrame({
    'Age': [40],
    'Gender': ['Female'],
    'Country': ['USA'],
    'Ethnicity': ['Caucasian'],
    'Family_History': ['No'],
    'Radiation_Exposure': ['No'],
    'Iodine_Deficiency': ['No'],
    'Smoking': ['No'],
    'Obesity': ['No'],
    'Diabetes': ['No'],
    'TSH_Level': [2.5],
    'T3_Level': [120],
    'T4_Level': [8.5],
    'Nodule_Size': [1.2],
    'Thyroid_Cancer_Risk': ['Low']
})

# Make prediction
predictions, probabilities = predictor.predict(new_patient_data)
print(f"Prediction: {predictions[0]}")
print(f"Probability of Malignant: {probabilities[0][1]:.3f}")
```

## Repository Structure
```
├── FinalProject.ipynb          # Main Jupyter notebook with implementation
├── thyroid_cancer_risk_data.csv # Dataset
├── README.md                    # Project documentation
└── images/                      # Visualizations and results
```

## Future Work
Future enhancements to this project could include:
* Improving model sensitivity to malignant cases
* Adding more advanced feature engineering
* Incorporating medical imaging data
* Creating a user-friendly web interface for clinicians

## Author
Hamid Shah

## Acknowledgments
Special thanks to the course instructor and teaching staff of CS 439 for their guidance and support throughout this project.
