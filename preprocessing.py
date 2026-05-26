"""
Module for data preprocessing and cleaning.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np


def preprocess_data(data):
    """
    Perform data preprocessing including:
    - Handling missing values
    - Encoding categorical variables
    - Feature scaling
    
    Args:
        data (pd.DataFrame): Raw dataset
        
    Returns:
        tuple: (X, y) where X is features and y is target variable
    """
    # Create a copy to avoid modifying original data
    df = data.copy()
    
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    # 1. Handle missing values
    print("\n1. Handling missing values...")
    df = df.dropna()
    print(f"   ✓ Removed rows with missing values. Shape: {df.shape}")
    
    # 2. Remove duplicates
    print("\n2. Removing duplicates...")
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    print(f"   ✓ Removed {removed} duplicate rows. Shape: {df.shape}")
    
    # 3. Separate features and target
    print("\n3. Separating features and target variable...")
    X = df.drop('Selling_Price', axis=1)
    y = df['Selling_Price']
    print(f"   ✓ Target variable shape: {y.shape}")
    print(f"   ✓ Features shape: {X.shape}")
    
    # 4. Handle categorical variables
    print("\n4. Encoding categorical variables...")
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"   Categorical columns: {categorical_cols}")
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"   ✓ Encoded '{col}': {len(le.classes_)} unique values")
    
    # 5. Feature scaling (normalization)
    print("\n5. Scaling numerical features...")
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    print(f"   ✓ Scaled {len(numerical_cols)} numerical features")
    
    print("\n6. Final preprocessed data shape:")
    print(f"   X: {X.shape}, y: {y.shape}")
    
    return X, y, label_encoders, scaler


def get_feature_names(data):
    """
    Get feature names after dropping target variable.
    
    Args:
        data (pd.DataFrame): Original dataset
        
    Returns:
        list: Feature names
    """
    return [col for col in data.columns if col != 'Selling_Price']


if __name__ == "__main__":
    from data_loader import load_data
    
    filepath = "dataset/car data.csv"
    data = load_data(filepath)
    X, y, label_encoders, scaler = preprocess_data(data)
    
    print("\nFeature columns:", X.columns.tolist())
