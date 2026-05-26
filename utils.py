"""
Utility functions and helper methods for the car price prediction project.
"""

import os
import json
from datetime import datetime


def create_directory_structure():
    """Create necessary directories if they don't exist."""
    directories = ['plots', 'models', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directory '{directory}' created/verified")


def save_configuration(config, filename='config.json'):
    """
    Save configuration to JSON file.
    
    Args:
        config (dict): Configuration dictionary
        filename (str): Output filename
    """
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✓ Configuration saved to {filename}")


def load_configuration(filename='config.json'):
    """
    Load configuration from JSON file.
    
    Args:
        filename (str): Configuration filename
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(filename):
        print(f"Warning: Configuration file '{filename}' not found")
        return {}
    
    with open(filename, 'r') as f:
        config = json.load(f)
    print(f"✓ Configuration loaded from {filename}")
    return config


def get_default_config():
    """
    Get default configuration for the project.
    
    Returns:
        dict: Default configuration
    """
    config = {
        'project_name': 'Car Price Prediction',
        'dataset': {
            'path': 'dataset/car data.csv',
            'target_column': 'Selling_Price',
            'test_size': 0.2,
            'random_state': 42
        },
        'preprocessing': {
            'handle_missing': True,
            'remove_duplicates': True,
            'scale_features': True
        },
        'feature_engineering': {
            'create_age_features': True,
            'create_mileage_features': True,
            'create_interactions': True,
            'num_selected_features': 10
        },
        'models': {
            'linear_regression': {'enabled': True},
            'ridge_regression': {'enabled': True, 'alpha': 1.0},
            'lasso_regression': {'enabled': True, 'alpha': 0.1},
            'random_forest': {'enabled': True, 'n_estimators': 100},
            'gradient_boosting': {'enabled': True, 'n_estimators': 100},
            'svr': {'enabled': True, 'kernel': 'rbf'}
        },
        'evaluation': {
            'cross_val_folds': 5,
            'metrics': ['r2', 'rmse', 'mae', 'mse']
        },
        'visualization': {
            'save_plots': True,
            'plot_format': 'png',
            'dpi': 300
        }
    }
    return config


def log_message(message, level='INFO', log_file='logs/project.log'):
    """
    Log a message to file and console.
    
    Args:
        message (str): Message to log
        level (str): Log level (INFO, WARNING, ERROR)
        log_file (str): Path to log file
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {level}: {message}"
    
    print(log_entry)
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a') as f:
        f.write(log_entry + '\n')


def print_metrics_table(results, title="Model Performance Metrics"):
    """
    Print results in a formatted table.
    
    Args:
        results (dict): Model evaluation results
        title (str): Table title
    """
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80)
    
    print(f"\n{'Model':<25} {'R² Score':<15} {'RMSE':<15} {'MAE':<15}")
    print("-"*80)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['R2']:<15.4f} "
              f"{metrics['RMSE']:<15.4f} {metrics['MAE']:<15.4f}")
    
    print("="*80 + "\n")


def validate_dataset(filepath):
    """
    Validate if dataset file exists and is readable.
    
    Args:
        filepath (str): Path to dataset
        
    Returns:
        bool: True if dataset is valid, False otherwise
    """
    if not os.path.exists(filepath):
        print(f"❌ Error: Dataset not found at {filepath}")
        return False
    
    if not filepath.endswith('.csv'):
        print(f"❌ Error: Invalid file format. Expected .csv, got {filepath}")
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            f.readline()
        print(f"✓ Dataset validation passed: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error reading dataset: {str(e)}")
        return False


def check_dependencies():
    """
    Check if all required dependencies are installed.
    
    Returns:
        bool: True if all dependencies are available
    """
    required_packages = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn', 'scipy']
    missing_packages = []
    
    print("\nChecking dependencies...")
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them using: pip install -r requirements.txt")
        return False
    else:
        print("✓ All dependencies are installed!")
        return True


def print_project_info():
    """Print project information and details."""
    info = """
    ╔════════════════════════════════════════════════════════════════╗
    ║           CAR PRICE PREDICTION - ML PROJECT                    ║
    ║                                                                ║
    ║  A comprehensive machine learning pipeline for predicting      ║
    ║  car prices using regression models and advanced features.    ║
    ║                                                                ║
    ║  Technologies:                                                 ║
    ║  - Python 3.7+                                                ║
    ║  - Pandas, NumPy for data manipulation                         ║
    ║  - Scikit-learn for ML models                                  ║
    ║  - Matplotlib, Seaborn for visualization                       ║
    ║                                                                ║
    ║  Models Included:                                              ║
    ║  1. Linear Regression                                          ║
    ║  2. Ridge Regression                                           ║
    ║  3. Lasso Regression                                           ║
    ║  4. Random Forest                                              ║
    ║  5. Gradient Boosting                                          ║
    ║  6. Support Vector Regression                                  ║
    ║                                                                ║
    ║  Run: python main.py                                           ║
    ╚════════════════════════════════════════════════════════════════╝
    """
    print(info)


if __name__ == "__main__":
    print_project_info()
    create_directory_structure()
    check_dependencies()
    
    # Show default configuration
    config = get_default_config()
    print("\nDefault Configuration:")
    print(json.dumps(config, indent=2))
