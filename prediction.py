"""
Module for making predictions on new car data using trained models.
"""

import pandas as pd
import numpy as np
import pickle
import os


def load_model(model_path):
    """
    Load a trained model from file.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        Trained model object
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✓ Model loaded from: {model_path}")
    return model


def predict_single_car(model, features_dict, feature_names, scaler):
    """
    Make prediction for a single car.
    
    Args:
        model: Trained model
        features_dict (dict): Dictionary with feature values
        feature_names (list): List of feature names in correct order
        scaler: Fitted scaler for normalization
        
    Returns:
        float: Predicted price
    """
    # Create array with features in correct order
    feature_values = np.array([features_dict.get(name, 0) for name in feature_names]).reshape(1, -1)
    
    # Scale features using the same scaler
    feature_values_scaled = scaler.transform(feature_values)
    
    # Make prediction
    prediction = model.predict(feature_values_scaled)[0]
    
    return prediction


def predict_batch(model, data_df, feature_names, scaler):
    """
    Make predictions for multiple cars.
    
    Args:
        model: Trained model
        data_df (pd.DataFrame): DataFrame with features
        feature_names (list): List of feature names
        scaler: Fitted scaler for normalization
        
    Returns:
        np.array: Array of predictions
    """
    # Extract features in correct order
    X = data_df[feature_names]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    return predictions


def create_sample_predictions():
    """
    Create sample predictions to demonstrate the prediction functionality.
    Requires a trained model and scaler to be available.
    """
    print("\n" + "="*60)
    print("SAMPLE CAR PRICE PREDICTIONS")
    print("="*60)
    
    # Sample car data (these are example values)
    sample_cars = pd.DataFrame({
        'Car_Name': ['Swift', 'Fortuner', 'City', 'Polo'],
        'Year': [2018, 2015, 2019, 2017],
        'Present_Price': [6.5, 15.5, 9.5, 7.5],
        'Driven_kms': [25000, 50000, 15000, 35000],
        'Fuel_Type': ['Petrol', 'Diesel', 'Petrol', 'Petrol'],
        'Selling_type': ['Dealer', 'Individual', 'Dealer', 'Dealer'],
        'Transmission': ['Manual', 'Manual', 'Automatic', 'Manual'],
        'Owner': [0, 1, 0, 0]
    })
    
    print("\nSample Cars for Prediction:")
    print(sample_cars)
    
    print("\nNote: To make actual predictions, run:")
    print("1. python main.py  # to train models")
    print("2. Use load_model() to load trained model")
    print("3. Call predict_batch() with new car data")


def interactive_prediction():
    """
    Interactive prediction interface for single car.
    Prompts user to input car features.
    """
    print("\n" + "="*60)
    print("INTERACTIVE CAR PRICE PREDICTION")
    print("="*60)
    
    print("\nEnter car details (or press Enter for default values):")
    
    car_features = {}
    
    try:
        car_features['Year'] = int(input("Car Year (e.g., 2018): ") or "2018")
        car_features['Present_Price'] = float(input("Present Price in millions (e.g., 6.5): ") or "6.5")
        car_features['Driven_kms'] = int(input("Kilometers driven (e.g., 25000): ") or "25000")
        
        print("\nFuel Type: 1=Petrol, 2=Diesel, 3=CNG")
        fuel_choice = int(input("Select fuel type (1-3): ") or "1")
        fuel_map = {1: 'Petrol', 2: 'Diesel', 3: 'CNG'}
        car_features['Fuel_Type'] = fuel_map.get(fuel_choice, 'Petrol')
        
        print("\nSelling Type: 1=Dealer, 2=Individual")
        selling_choice = int(input("Select selling type (1-2): ") or "1")
        selling_map = {1: 'Dealer', 2: 'Individual'}
        car_features['Selling_type'] = selling_map.get(selling_choice, 'Dealer')
        
        print("\nTransmission: 1=Manual, 2=Automatic")
        trans_choice = int(input("Select transmission (1-2): ") or "1")
        trans_map = {1: 'Manual', 2: 'Automatic'}
        car_features['Transmission'] = trans_map.get(trans_choice, 'Manual')
        
        car_features['Owner'] = int(input("Number of previous owners (0-3): ") or "0")
        
        print("\nEntered Car Details:")
        for key, value in car_features.items():
            print(f"  {key}: {value}")
        
        return car_features
    
    except ValueError:
        print("❌ Invalid input. Please enter valid values.")
        return None


def format_prediction_output(car_features, prediction):
    """
    Format and display prediction results.
    
    Args:
        car_features (dict): Input car features
        prediction (float): Predicted price
    """
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    
    print("\nCar Details:")
    for key, value in car_features.items():
        print(f"  {key}: {value}")
    
    print(f"\n  💰 Predicted Selling Price: ₹{prediction:.2f} Million")
    print(f"     (Approximately ₹{prediction*10:.0f} Lakhs)")
    
    # Show confidence range (assuming ±15% uncertainty)
    lower_bound = prediction * 0.85
    upper_bound = prediction * 1.15
    print(f"\n  Price Range (±15% uncertainty):")
    print(f"    - Lower: ₹{lower_bound:.2f} Million")
    print(f"    - Upper: ₹{upper_bound:.2f} Million")


def demonstrate_prediction_workflow():
    """
    Demonstrate the complete prediction workflow.
    """
    print("\n" + "="*60)
    print("PREDICTION WORKFLOW DEMONSTRATION")
    print("="*60)
    
    workflow = """
    STEP 1: Train Models
    ====================
    Run: python main.py
    This will train and save the best model
    
    STEP 2: Load Model
    ==================
    model = load_model('models/best_model_*.pkl')
    
    STEP 3: Prepare Data
    ====================
    # For single prediction:
    features = {
        'Year': 2018,
        'Present_Price': 6.5,
        'Driven_kms': 25000,
        'Fuel_Type': 'Petrol',
        'Selling_type': 'Dealer',
        'Transmission': 'Manual',
        'Owner': 0
    }
    
    # For batch prediction:
    new_data = pd.read_csv('new_cars.csv')
    
    STEP 4: Make Prediction
    =======================
    # Single car:
    price = predict_single_car(model, features, feature_names, scaler)
    
    # Batch prediction:
    prices = predict_batch(model, new_data, feature_names, scaler)
    
    STEP 5: Display Results
    =======================
    format_prediction_output(features, price)
    """
    
    print(workflow)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CAR PRICE PREDICTION - INFERENCE MODULE")
    print("="*60)
    
    create_sample_predictions()
    demonstrate_prediction_workflow()
    
    print("\nNote: This module is imported by other scripts for predictions.")
    print("To use this module:")
    print("  from prediction import load_model, predict_single_car")
