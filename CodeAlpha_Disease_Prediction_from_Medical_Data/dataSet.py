import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_breast_cancer, load_diabetes

print("--- Task 4: Disease Prediction from Medical Data ---")
print("--- Step 1: Data Acquisition & Initial Exploration (Heart Disease Dataset) ---")

def load_heart_disease():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'dcol', 'fbs', 'restecg', 'thalach',
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
    ]
    df = pd.read_csv(url, names=column_names, na_values='?').dropna()
    df['num'] = (df['num'] > 0).astype(int)
    X = df.drop('num', axis=1)
    y = df['num']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y.values, scaler, encoders, X.columns.tolist()

def load_breast_cancer_data():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y.values, scaler, {}, X.columns.tolist()

def load_diabetes_data():
    data = load_diabetes(as_frame=True)
    X = data.data
    y = (data.target > data.target.median()).astype(int)  # Binarize for classification
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y.values, scaler, {}, X.columns.tolist()

def load_and_preprocess_data(disease):
    if disease == 'Heart Disease':
        return load_heart_disease()
    elif disease == 'Breast Cancer':
        return load_breast_cancer_data()
    elif disease == 'Diabetes':
        return load_diabetes_data()
    else:
        raise ValueError('Unknown disease type')

try:
    # Load the dataset
    # 'na_values='?' handles missing values represented by '?'
    X_scaled, y, scaler, encoders, feature_names = load_and_preprocess_data('Heart Disease')

    print("\nDataset loaded successfully!")
    print(f"Shape of the dataset: {X_scaled.shape}")

    print("\n--- First 5 rows of the dataset: ---")
    # Convert scaled data back to DataFrame for easier viewing
    df_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    print(df_scaled.head())

    print("\n--- Dataset Info (Column types and non-null counts): ---")
    df_scaled.info()

    print("\n--- Descriptive Statistics: ---")
    print(df_scaled.describe())

    print("\n--- Value Counts for the Target Variable ('num'): ---")
    # 'num' is the target variable: 0 = no heart disease, >0 = heart disease
    # We will later convert this to a binary 0 or 1.
    print(pd.Series(y).value_counts())

    print("\n--- Check for Missing Values: ---")
    print(df_scaled.isnull().sum())

except Exception as e:
    print(f"Error loading the dataset: {e}")
    print("Please check your internet connection or the URL for the dataset.")
    print("You might need to download the dataset manually and specify its local path.")