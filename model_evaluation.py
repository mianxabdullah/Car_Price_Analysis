"""
Module for evaluating model performance.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_models(models, X_test, y_test):
    """
    Evaluate all trained models on test set.
    
    Args:
        models (dict): Dictionary of trained models
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target values
        
    Returns:
        dict: Evaluation metrics for each model
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION ON TEST SET")
    print("="*60)
    
    results = {}
    
    for model_name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[model_name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Predictions': y_pred
        }
        
        print(f"\n{model_name}:")
        print(f"  Mean Squared Error (MSE):    {mse:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"  Mean Absolute Error (MAE):   {mae:.4f}")
        print(f"  R² Score:                    {r2:.4f}")
    
    return results


def compare_models(results):
    """
    Compare all models and identify the best one.
    
    Args:
        results (dict): Evaluation results from evaluate_models()
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Sort models by R² score (higher is better)
    sorted_models = sorted(results.items(), 
                          key=lambda x: x[1]['R2'], 
                          reverse=True)
    
    print("\nModels ranked by R² Score (Higher is Better):\n")
    print(f"{'Rank':<6} {'Model':<25} {'R² Score':<12} {'RMSE':<12} {'MAE':<12}")
    print("-" * 65)
    
    for rank, (model_name, metrics) in enumerate(sorted_models, 1):
        print(f"{rank:<6} {model_name:<25} {metrics['R2']:<12.4f} "
              f"{metrics['RMSE']:<12.4f} {metrics['MAE']:<12.4f}")
    
    best_model_name = sorted_models[0][0]
    print(f"\n✓ Best Model: {best_model_name}")
    print(f"  R² Score: {sorted_models[0][1]['R2']:.4f}")
    
    return best_model_name, sorted_models[0][1]


def calculate_prediction_error(y_test, y_pred):
    """
    Calculate prediction errors for detailed analysis.
    
    Args:
        y_test (pd.Series): Actual values
        y_pred (np.array): Predicted values
        
    Returns:
        dict: Error statistics
    """
    errors = y_test - y_pred
    
    error_stats = {
        'Mean_Error': errors.mean(),
        'Std_Error': errors.std(),
        'Min_Error': errors.min(),
        'Max_Error': errors.max(),
        'Median_Error': errors.median() if hasattr(errors, 'median') else np.median(errors),
        'Mean_Abs_Error': np.abs(errors).mean()
    }
    
    return error_stats


def print_error_analysis(y_test, y_pred, model_name):
    """
    Print detailed error analysis.
    
    Args:
        y_test (pd.Series): Actual values
        y_pred (np.array): Predicted values
        model_name (str): Name of the model
    """
    print("\n" + "="*60)
    print(f"ERROR ANALYSIS: {model_name}")
    print("="*60)
    
    error_stats = calculate_prediction_error(y_test, y_pred)
    
    print("\nPrediction Error Statistics:")
    print(f"  Mean Error:          {error_stats['Mean_Error']:.4f}")
    print(f"  Std Dev of Errors:   {error_stats['Std_Error']:.4f}")
    print(f"  Min Error:           {error_stats['Min_Error']:.4f}")
    print(f"  Max Error:           {error_stats['Max_Error']:.4f}")
    print(f"  Median Error:        {error_stats['Median_Error']:.4f}")
    print(f"  Mean Abs Error:      {error_stats['Mean_Abs_Error']:.4f}")
    
    # Calculate percentage error
    percentage_errors = np.abs((y_test - y_pred) / y_test) * 100
    print(f"\n  Mean Percentage Error: {percentage_errors.mean():.2f}%")
    print(f"  Std Dev Percentage Error: {percentage_errors.std():.2f}%")


def get_model_summary(models, results):
    """
    Generate summary of all models.
    
    Args:
        models (dict): Dictionary of trained models
        results (dict): Evaluation results
    """
    print("\n" + "="*60)
    print("FINAL MODEL SUMMARY")
    print("="*60)
    
    print(f"\nTotal Models Trained: {len(models)}")
    print("\nModel Details:")
    
    for model_name, metrics in sorted(results.items(), 
                                      key=lambda x: x[1]['R2'], 
                                      reverse=True):
        print(f"\n  {model_name}")
        print(f"    - R² Score: {metrics['R2']:.4f}")
        print(f"    - RMSE: {metrics['RMSE']:.4f}")
        print(f"    - MAE: {metrics['MAE']:.4f}")


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preprocess_data
    from feature_engineering import engineer_features, select_best_features
    from model_training import split_data, train_models
    
    filepath = "dataset/car data.csv"
    data = load_data(filepath)
    X, y, _, _ = preprocess_data(data)
    X_engineered = engineer_features(X, y)
    X_selected, _ = select_best_features(X_engineered, y, k=10)
    
    X_train, X_test, y_train, y_test = split_data(X_selected, y)
    models = train_models(X_train, y_train)
    
    results = evaluate_models(models, X_test, y_test)
    best_model_name, best_metrics = compare_models(results)
    get_model_summary(models, results)
