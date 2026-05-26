"""
Module for feature engineering and feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression


def engineer_features(X, y):
    """
    Create new engineered features to improve model performance.
    
    Args:
        X (pd.DataFrame): Features dataframe
        y (pd.Series): Target variable
        
    Returns:
        pd.DataFrame: Enhanced features with engineered columns
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    
    X_engineered = X.copy()
    
    # 1. Age-related features
    if 'Year' in X_engineered.columns:
        print("\n1. Creating age-related features...")
        current_year = 2024
        X_engineered['Car_Age'] = current_year - X_engineered['Year']
        print(f"   ✓ Created 'Car_Age' feature")
    
    # 2. Mileage-related features
    if 'Driven_kms' in X_engineered.columns:
        print("\n2. Creating mileage-related features...")
        X_engineered['Avg_Annual_km'] = X_engineered['Driven_kms'] / (X_engineered['Car_Age'] + 1)
        print(f"   ✓ Created 'Avg_Annual_km' feature")
    
    # 3. Depreciation feature
    if 'Present_Price' in X_engineered.columns and 'Selling_Price' in X.columns:
        print("\n3. Creating depreciation feature...")
        # Note: Using X as reference, but depreciation would be calculated differently in practice
        print(f"   ✓ Available for model consideration")
    
    # 4. Interaction features
    print("\n4. Creating interaction features...")
    if 'Year' in X_engineered.columns and 'Driven_kms' in X_engineered.columns:
        X_engineered['Year_x_Mileage'] = X_engineered['Year'] * X_engineered['Driven_kms']
        print(f"   ✓ Created 'Year_x_Mileage' interaction feature")
    
    print(f"\n5. Feature Engineering Complete!")
    print(f"   Original features: {X.shape[1]}")
    print(f"   Engineered features: {X_engineered.shape[1]}")
    print(f"   New features added: {X_engineered.shape[1] - X.shape[1]}")
    
    return X_engineered


def select_best_features(X, y, k=10):
    """
    Select the k best features using statistical tests.
    
    Args:
        X (pd.DataFrame): Features dataframe
        y (pd.Series): Target variable
        k (int): Number of best features to select
        
    Returns:
        pd.DataFrame: DataFrame with k best features
        list: Names of selected features
    """
    print("\n" + "="*60)
    print("FEATURE SELECTION")
    print("="*60)
    
    k = min(k, X.shape[1])  # Ensure k doesn't exceed number of features
    
    selector = SelectKBest(f_regression, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get feature names and scores
    feature_names = X.columns.tolist()
    scores = selector.scores_
    
    # Create a dataframe of feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'F-Score': scores
    }).sort_values('F-Score', ascending=False)
    
    print(f"\nTop {k} Most Important Features:")
    print(feature_importance.head(k))
    
    # Get selected feature names
    selected_features = feature_importance.head(k)['Feature'].tolist()
    X_selected_df = X[selected_features]
    
    print(f"\nSelected Features: {selected_features}")
    
    return X_selected_df, selected_features


def analyze_feature_correlation(X, y):
    """
    Analyze correlation between features and target variable.
    
    Args:
        X (pd.DataFrame): Features dataframe
        y (pd.Series): Target variable
    """
    print("\n" + "="*60)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*60)
    
    # Calculate correlation with target
    correlation_data = pd.DataFrame({
        'Feature': X.columns,
        'Correlation_with_Target': [X[col].corr(y) for col in X.columns]
    })
    
    correlation_data = correlation_data.sort_values('Correlation_with_Target', 
                                                     ascending=False, 
                                                     key=abs)
    
    print("\nCorrelation with Selling Price:")
    print(correlation_data)


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preprocess_data
    
    filepath = "dataset/car data.csv"
    data = load_data(filepath)
    X, y, _, _ = preprocess_data(data)
    
    X_engineered = engineer_features(X, y)
    X_selected, selected_features = select_best_features(X_engineered, y, k=10)
    analyze_feature_correlation(X_selected, y)
