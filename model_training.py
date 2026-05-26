"""
Module for training regression models.
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import pickle


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n" + "="*60)
    print("TRAIN-TEST SPLIT")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    print(f"Train-Test ratio: {X_train.shape[0]/len(X):.1%} - {test_size:.1%}")
    
    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """
    Train multiple regression models.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        dict: Dictionary of trained models
    """
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    models = {}
    
    # 1. Linear Regression
    print("\n1. Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['Linear Regression'] = lr
    print("   ✓ Linear Regression trained")
    
    # 2. Ridge Regression
    print("\n2. Training Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    models['Ridge Regression'] = ridge
    print("   ✓ Ridge Regression trained (alpha=1.0)")
    
    # 3. Lasso Regression
    print("\n3. Training Lasso Regression...")
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    models['Lasso Regression'] = lasso
    print("   ✓ Lasso Regression trained (alpha=0.1)")
    
    # 4. Random Forest
    print("\n4. Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    print("   ✓ Random Forest trained (100 estimators)")
    
    # 5. Gradient Boosting
    print("\n5. Training Gradient Boosting Regressor...")
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    print("   ✓ Gradient Boosting trained (100 estimators)")
    
    # 6. Support Vector Regression
    print("\n6. Training Support Vector Regression...")
    svr = SVR(kernel='rbf', C=100, epsilon=0.1)
    svr.fit(X_train, y_train)
    models['SVR'] = svr
    print("   ✓ SVR trained (kernel=rbf)")
    
    print(f"\n✓ Total models trained: {len(models)}")
    
    return models


def evaluate_with_cross_validation(X_train, y_train, models, cv=5):
    """
    Evaluate models using cross-validation.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        models (dict): Dictionary of models
        cv (int): Number of cross-validation folds
    """
    print("\n" + "="*60)
    print("CROSS-VALIDATION EVALUATION")
    print("="*60)
    
    cv_results = {}
    
    for model_name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, 
                                cv=cv, scoring='r2')
        cv_results[model_name] = scores
        
        print(f"\n{model_name}:")
        print(f"  CV Scores: {scores}")
        print(f"  Mean R² Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    return cv_results


def save_model(model, filepath):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        filepath (str): Path to save the model
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {filepath}")


def load_model(filepath):
    """
    Load trained model from disk.
    
    Args:
        filepath (str): Path to saved model
        
    Returns:
        Loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model loaded from {filepath}")
    return model


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preprocess_data
    from feature_engineering import engineer_features, select_best_features
    
    filepath = "dataset/car data.csv"
    data = load_data(filepath)
    X, y, _, _ = preprocess_data(data)
    X_engineered = engineer_features(X, y)
    X_selected, _ = select_best_features(X_engineered, y, k=10)
    
    X_train, X_test, y_train, y_test = split_data(X_selected, y)
    models = train_models(X_train, y_train)
    evaluate_with_cross_validation(X_train, y_train, models)
