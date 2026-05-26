"""
Module for loading and exploring the car dataset.
"""

import pandas as pd
import os


def load_data(filepath):
    """
    Load the car dataset from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    data = pd.read_csv(filepath)
    print(f"✓ Dataset loaded successfully!")
    print(f"  Shape: {data.shape}")
    return data


def explore_data(data):
    """
    Perform basic exploratory data analysis.
    
    Args:
        data (pd.DataFrame): Dataset to explore
    """
    print("\n" + "="*60)
    print("DATA OVERVIEW")
    print("="*60)
    
    print("\n1. First few rows:")
    print(data.head())
    
    print("\n2. Dataset Info:")
    print(data.info())
    
    print("\n3. Statistical Summary:")
    print(data.describe())
    
    print("\n4. Missing Values:")
    missing = data.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values found!")
    else:
        print(missing[missing > 0])
    
    print("\n5. Data Types:")
    print(data.dtypes)
    
    print("\n6. Target Variable Distribution (Selling_Price):")
    print(f"  Mean: {data['Selling_Price'].mean():.2f}")
    print(f"  Median: {data['Selling_Price'].median():.2f}")
    print(f"  Min: {data['Selling_Price'].min():.2f}")
    print(f"  Max: {data['Selling_Price'].max():.2f}")
    print(f"  Std: {data['Selling_Price'].std():.2f}")


if __name__ == "__main__":
    filepath = "dataset/car data.csv"
    data = load_data(filepath)
    explore_data(data)
