"""
Main script to run the entire car price prediction pipeline.
This script orchestrates the machine learning workflow from data loading to model evaluation.
"""

from data_loader import load_data, explore_data
from preprocessing import preprocess_data
from feature_engineering import engineer_features, select_best_features, analyze_feature_correlation
from model_training import split_data, train_models, evaluate_with_cross_validation, save_model
from model_evaluation import evaluate_models, compare_models, print_error_analysis, get_model_summary
from visualization import (
    set_style, create_output_directory, plot_price_distribution,
    plot_feature_correlation, plot_actual_vs_predicted,
    plot_residuals, plot_model_comparison, plot_feature_importance
)
from utils import check_dependencies


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {title.center(66)}")
    print("="*70)


def main():
    """Main pipeline execution."""
    
    print_header("CAR PRICE PREDICTION - MACHINE LEARNING PIPELINE")
    
    # 0. Setup: Check dependencies and create directories
    print_header("STEP 0: SETUP AND VERIFICATION")
    check_dependencies()
    create_output_directory()
    set_style()
    
    # 1. Data Loading and Exploration
    print_header("STEP 1: DATA LOADING AND EXPLORATION")
    filepath = "dataset/car data.csv"
    data = load_data(filepath)
    explore_data(data)
    
    # 2. Data Preprocessing
    print_header("STEP 2: DATA PREPROCESSING")
    X, y, label_encoders, scaler = preprocess_data(data)
    
    # 3. Feature Engineering
    print_header("STEP 3: FEATURE ENGINEERING")
    X_engineered = engineer_features(X, y)
    analyze_feature_correlation(X_engineered, y)
    
    # 4. Feature Selection
    print_header("STEP 4: FEATURE SELECTION")
    X_selected, selected_features = select_best_features(X_engineered, y, k=10)
    
    # 5. Train-Test Split
    print_header("STEP 5: TRAIN-TEST SPLIT")
    X_train, X_test, y_train, y_test = split_data(X_selected, y)
    
    # 6. Model Training
    print_header("STEP 6: MODEL TRAINING")
    models = train_models(X_train, y_train)
    
    # 7. Cross-Validation
    print_header("STEP 7: CROSS-VALIDATION EVALUATION")
    cv_results = evaluate_with_cross_validation(X_train, y_train, models, cv=5)
    
    # 8. Model Evaluation
    print_header("STEP 8: TEST SET EVALUATION")
    results = evaluate_models(models, X_test, y_test)
    
    # 9. Model Comparison
    print_header("STEP 9: MODEL COMPARISON AND BEST MODEL SELECTION")
    best_model_name, best_metrics = compare_models(results)
    best_model = models[best_model_name]
    
    # 10. Error Analysis
    print_header("STEP 10: DETAILED ERROR ANALYSIS")
    best_predictions = results[best_model_name]['Predictions']
    print_error_analysis(y_test, best_predictions, best_model_name)
    
    # 11. Final Summary
    print_header("STEP 11: FINAL MODEL SUMMARY")
    get_model_summary(models, results)
    
    # 12. Visualizations
    print_header("STEP 12: GENERATING VISUALIZATIONS")
    
    print("\nGenerating plots...")
    plot_price_distribution(y)
    plot_feature_correlation(X_selected, y)
    plot_model_comparison(results)
    plot_actual_vs_predicted(y_test, best_predictions, best_model_name)
    plot_residuals(y_test, best_predictions, best_model_name)
    
    # Feature importance for tree-based models
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        plot_feature_importance(best_model, selected_features, best_model_name)
    
    # 13. Save Best Model
    print_header("STEP 13: SAVING BEST MODEL")
    model_path = f"models/best_model_{best_model_name.replace(' ', '_').lower()}.pkl"
    save_model(best_model, model_path)
    
    # 14. Final Report
    print_header("FINAL REPORT")
    print(f"""
    ✓ Pipeline Execution Complete!
    
    PROJECT SUMMARY:
    ================
    
    Dataset Information:
    - Total samples: {len(data)}
    - Training samples: {len(X_train)}
    - Testing samples: {len(X_test)}
    - Number of features (engineered): {X_engineered.shape[1]}
    - Selected features: {len(selected_features)}
    
    Models Trained: {len(models)}
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - Random Forest
    - Gradient Boosting
    - Support Vector Regression
    
    BEST MODEL: {best_model_name}
    ===============================
    - R² Score: {best_metrics['R2']:.4f}
    - RMSE: {best_metrics['RMSE']:.4f}
    - MAE: {best_metrics['MAE']:.4f}
    
    OUTPUT FILES:
    - Plots saved in: plots/
    - Model saved in: {model_path}
    
    NEXT STEPS:
    - Review the plots in the plots/ folder
    - Use load_model() to load the saved best model
    - Deploy the model for price predictions
    
    """)


if __name__ == "__main__":
    try:
        main()
        print("\n" + "="*70)
        print("✓ PROJECT COMPLETED SUCCESSFULLY!".center(70))
        print("="*70 + "\n")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("Please check the error message above and try again.")
        raise
