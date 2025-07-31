import streamlit as st
import numpy as np
from dataSet import load_and_preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

st.title("Disease Prediction from Medical Data")
st.write("Predict the possibility of diseases based on patient data. Select a disease and model to begin.")

# Disease and model selection
DISEASES = ['Heart Disease', 'Breast Cancer', 'Diabetes']
DISEASE_DESCRIPTIONS = {
    'Heart Disease': "Cleveland Heart Disease dataset. Features: age, sex, chest pain, blood pressure, cholesterol, etc.",
    'Breast Cancer': "Breast Cancer Wisconsin (Diagnostic) dataset. Features: cell nuclei measurements from breast mass.",
    'Diabetes': "Diabetes dataset (UCI/Scikit-learn). Features: age, sex, BMI, blood pressure, and blood serum measurements."
}
DISEASE_FEATURES = {
    'Heart Disease': [
        ('age', 'Age in years'),
        ('sex', 'Sex (0 = female, 1 = male)'),
        ('cp', 'Chest pain type (0-3)'),
        ('trestbps', 'Resting blood pressure'),
        ('dcol', 'Serum cholesterol'),
        ('fbs', 'Fasting blood sugar > 120 mg/dl (0/1)'),
        ('restecg', 'Resting ECG results (0-2)'),
        ('thalach', 'Max heart rate achieved'),
        ('exang', 'Exercise induced angina (0/1)'),
        ('oldpeak', 'ST depression induced by exercise'),
        ('slope', 'Slope of peak exercise ST segment (0-2)'),
        ('ca', 'Number of major vessels (0-3)'),
        ('thal', 'Thalassemia (1,2,3)')
    ],
    'Breast Cancer': [(f, '') for f in [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
        'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
        'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
        'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']
    ],
    'Diabetes': [(f, '') for f in [
        'age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']]
}
MODELS = {
    'Random Forest': RandomForestClassifier,
    'Logistic Regression': LogisticRegression,
    'SVM': SVC,
    'XGBoost': XGBClassifier
}
MODEL_DESCRIPTIONS = {
    'Random Forest': 'Ensemble of decision trees, robust to overfitting.',
    'Logistic Regression': 'Linear model for binary classification.',
    'SVM': 'Support Vector Machine, effective in high-dimensional spaces.',
    'XGBoost': 'Gradient boosting, powerful for tabular data.'
}

disease = st.sidebar.selectbox('Select Disease', DISEASES)
st.sidebar.info(DISEASE_DESCRIPTIONS[disease])
model_name = st.sidebar.selectbox('Select Model', list(MODELS.keys()))
st.sidebar.caption(MODEL_DESCRIPTIONS[model_name])

with st.expander("Show feature explanations"):
    for feat, desc in DISEASE_FEATURES[disease]:
        st.write(f"**{feat}**: {desc}")

# Load and preprocess data for selected disease
X, y, scaler, encoders, feature_names = load_and_preprocess_data(disease)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model instantiation
if model_name == 'SVM':
    model = MODELS[model_name](probability=True, random_state=42)
elif model_name == 'Logistic Regression':
    model = MODELS[model_name](max_iter=1000, random_state=42)
else:
    model = MODELS[model_name](random_state=42)
model.fit(X_train, y_train)

# Dynamic input UI for features
def user_input_features(feature_names):
    input_data = []
    st.sidebar.header('Patient Data')
    for feature in feature_names:
        if feature in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
            # Categorical features for heart disease
            if feature == 'sex':
                input_data.append(st.sidebar.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female'))
            elif feature == 'cp':
                input_data.append(st.sidebar.selectbox('Chest Pain Type (cp)', options=[0, 1, 2, 3]))
            elif feature == 'fbs':
                input_data.append(st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=[0, 1]))
            elif feature == 'restecg':
                input_data.append(st.sidebar.selectbox('Resting ECG (restecg)', options=[0, 1, 2]))
            elif feature == 'exang':
                input_data.append(st.sidebar.selectbox('Exercise Induced Angina (exang)', options=[0, 1]))
            elif feature == 'slope':
                input_data.append(st.sidebar.selectbox('Slope of Peak Exercise ST (slope)', options=[0, 1, 2]))
            elif feature == 'ca':
                input_data.append(st.sidebar.selectbox('Number of Major Vessels (ca)', options=[0, 1, 2, 3]))
            elif feature == 'thal':
                input_data.append(st.sidebar.selectbox('Thal', options=[1, 2, 3]))
        else:
            # Numeric features
            if 'age' in feature:
                input_data.append(st.sidebar.number_input('Age', min_value=10, max_value=100, value=50))
            elif 'trestbps' in feature:
                input_data.append(st.sidebar.number_input('Resting Blood Pressure (trestbps)', min_value=80, max_value=200, value=120))
            elif 'dcol' in feature:
                input_data.append(st.sidebar.number_input('Serum Cholesterol (dcol)', min_value=100, max_value=600, value=200))
            elif 'thalach' in feature:
                input_data.append(st.sidebar.number_input('Max Heart Rate (thalach)', min_value=60, max_value=220, value=150))
            elif 'oldpeak' in feature:
                input_data.append(st.sidebar.number_input('Oldpeak (ST depression)', min_value=0.0, max_value=10.0, value=1.0, step=0.1))
            else:
                # For other features, use a generic number input
                input_data.append(st.sidebar.number_input(feature, value=0.0))
    return np.array(input_data).reshape(1, -1)

user_input = user_input_features(feature_names)

def preprocess_user_input(user_input, scaler):
    user_input_scaled = scaler.transform(user_input)
    return user_input_scaled

if st.sidebar.button('Predict'):
    user_input_scaled = preprocess_user_input(user_input, scaler)
    prediction = model.predict(user_input_scaled)[0]
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(user_input_scaled)[0][1]
    else:
        probability = None
    st.subheader('Prediction Result:')
    if prediction == 1:
        if probability is not None:
            st.error(f'High risk ({probability*100:.1f}% probability)')
        else:
            st.error('High risk')
    else:
        if probability is not None:
            st.success(f'Low risk ({(1-probability)*100:.1f}% probability)')
        else:
            st.success('Low risk')

st.sidebar.markdown("---")
st.sidebar.write("Model accuracy on test set:")
st.sidebar.write(f"{model.score(X_test, y_test)*100:.2f}%") 