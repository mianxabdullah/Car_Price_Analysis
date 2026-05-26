"""
Module for visualization and plotting.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def set_style():
    """Set matplotlib and seaborn styling."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    plt.rcParams['font.size'] = 10


def plot_price_distribution(y):
    """
    Plot distribution of car prices.
    
    Args:
        y (pd.Series): Target variable (Selling_Price)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(y, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Selling Price')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Car Selling Prices')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(y, vert=True)
    axes[1].set_ylabel('Selling Price')
    axes[1].set_title('Box Plot of Car Selling Prices')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/price_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Price distribution plot saved to plots/price_distribution.png")


def plot_feature_correlation(X, y):
    """
    Plot correlation heatmap.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
    """
    # Add target to dataframe for correlation
    data = X.copy()
    data['Selling_Price'] = y
    
    # Select top numerical columns for better visualization
    if len(data.columns) > 15:
        # Calculate correlation with target and select top features
        correlations = data.corr()['Selling_Price'].abs().sort_values(ascending=False)
        top_features = correlations.head(10).index.tolist()
        data = data[top_features]
    
    # Create correlation matrix
    corr_matrix = data.corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('plots/feature_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Correlation heatmap saved to plots/feature_correlation.png")


def plot_actual_vs_predicted(y_test, y_pred, model_name):
    """
    Plot actual vs predicted values.
    
    Args:
        y_test (pd.Series): Actual test values
        y_pred (np.array): Predicted values
        model_name (str): Name of the model
    """
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(y_test, y_pred, alpha=0.6, s=50, color='blue', edgecolors='black')
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Selling Price')
    plt.ylabel('Predicted Selling Price')
    plt.title(f'{model_name}: Actual vs Predicted Prices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/actual_vs_predicted_{model_name.replace(" ", "_").lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Actual vs Predicted plot saved")


def plot_residuals(y_test, y_pred, model_name):
    """
    Plot residuals analysis.
    
    Args:
        y_test (pd.Series): Actual test values
        y_pred (np.array): Predicted values
        model_name (str): Name of the model
    """
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Residuals scatter plot
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=50, color='green', edgecolors='black')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted Values')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals histogram
    axes[0, 1].hist(residuals, bins=30, color='orange', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Residuals')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q plot (approximate)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Residuals over predictions
    axes[1, 1].plot(y_pred, np.abs(residuals), 'o', alpha=0.6, color='purple', markersize=5)
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Absolute Residuals')
    axes[1, 1].set_title('Absolute Residuals vs Predicted')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/residuals_{model_name.replace(" ", "_").lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Residuals plot saved")


def plot_model_comparison(results):
    """
    Plot comparison of all models.
    
    Args:
        results (dict): Evaluation results from evaluate_models()
    """
    model_names = list(results.keys())
    r2_scores = [results[m]['R2'] for m in model_names]
    rmse_scores = [results[m]['RMSE'] for m in model_names]
    mae_scores = [results[m]['MAE'] for m in model_names]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # R² Score
    bars1 = axes[0].barh(model_names, r2_scores, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('R² Score')
    axes[0].set_title('Model Comparison: R² Score (Higher is Better)')
    axes[0].set_xlim([0, 1])
    for i, v in enumerate(r2_scores):
        axes[0].text(v - 0.05, i, f'{v:.4f}', va='center', ha='right', color='white', fontweight='bold')
    
    # RMSE
    bars2 = axes[1].barh(model_names, rmse_scores, color='lightcoral', edgecolor='black')
    axes[1].set_xlabel('RMSE')
    axes[1].set_title('Model Comparison: RMSE (Lower is Better)')
    for i, v in enumerate(rmse_scores):
        axes[1].text(v, i, f'{v:.4f}', va='center', fontsize=9)
    
    # MAE
    bars3 = axes[2].barh(model_names, mae_scores, color='lightgreen', edgecolor='black')
    axes[2].set_xlabel('MAE')
    axes[2].set_title('Model Comparison: MAE (Lower is Better)')
    for i, v in enumerate(mae_scores):
        axes[2].text(v, i, f'{v:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Model comparison plot saved to plots/model_comparison.png")


def plot_feature_importance(model, feature_names, model_name):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model (must have feature_importances_ attribute)
        feature_names (list): Names of features
        model_name (str): Name of the model
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"Note: {model_name} does not support feature importance plotting")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15 features
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Top 15 Feature Importances - {model_name}')
    plt.bar(range(len(indices)), importances[indices], color='steelblue', edgecolor='black')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(f'plots/feature_importance_{model_name.replace(" ", "_").lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Feature importance plot saved")


def create_output_directory():
    """Create plots directory if it doesn't exist."""
    import os
    os.makedirs('plots', exist_ok=True)


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preprocess_data
    from feature_engineering import engineer_features, select_best_features
    from model_training import split_data, train_models
    from model_evaluation import evaluate_models
    
    set_style()
    create_output_directory()
    
    filepath = "dataset/car data.csv"
    data = load_data(filepath)
    X, y, _, _ = preprocess_data(data)
    X_engineered = engineer_features(X, y)
    X_selected, selected_features = select_best_features(X_engineered, y, k=10)
    
    plot_price_distribution(y)
    plot_feature_correlation(X_selected, y)
    
    X_train, X_test, y_train, y_test = split_data(X_selected, y)
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    
    plot_model_comparison(results)
