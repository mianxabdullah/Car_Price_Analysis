# Car Price Prediction with Machine Learning

A comprehensive machine learning project that predicts car prices using regression models. This project demonstrates the complete ML pipeline from data preprocessing to model evaluation and deployment.

## Project Overview

This project builds a robust price prediction system for cars by:
- Loading and exploring car-related data
- Performing data preprocessing and cleaning
- Engineering new features for better predictions
- Training multiple regression models
- Evaluating and comparing model performance
- Visualizing results and insights

## Dataset

**Source**: `dataset/car data.csv`

**Features**:
- `Car_Name`: Brand and model name
- `Year`: Year of manufacture
- `Present_Price`: Current market value (in millions)
- `Selling_Price`: Actual selling price (target variable) (in millions)
- `Driven_kms`: Total kilometers driven
- `Fuel_Type`: Type of fuel (Petrol, Diesel, CNG)
- `Selling_type`: Dealer or Individual
- `Transmission`: Manual or Automatic
- `Owner`: Number of previous owners

## Project Structure

```
Task-CarPrice/
├── dataset/
│   └── car data.csv                 # Input dataset
├── plots/                            # Generated visualization plots
├── models/                           # Saved trained models
├── data_loader.py                    # Data loading and exploration
├── preprocessing.py                  # Data preprocessing and cleaning
├── feature_engineering.py            # Feature engineering and selection
├── model_training.py                 # Model training and cross-validation
├── model_evaluation.py               # Model evaluation metrics
├── visualization.py                  # Plotting and visualization utilities
├── main.py                           # Main pipeline orchestrator
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Installation

### 1. Clone or Download the Project
```bash
cd Task-CarPrice
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Run the Complete Pipeline
Execute the main script to run the entire ML pipeline:
```bash
python main.py
```

This will:
1. Load and explore the dataset
2. Preprocess the data
3. Engineer new features
4. Select the best features
5. Train 6 different regression models
6. Evaluate models on test set
7. Compare model performance
8. Generate visualizations
9. Save the best model

### Run Individual Modules

#### 1. Data Loading and Exploration
```bash
python data_loader.py
```

#### 2. Data Preprocessing
```bash
python preprocessing.py
```

#### 3. Feature Engineering
```bash
python feature_engineering.py
```

#### 4. Model Training
```bash
python model_training.py
```

#### 5. Model Evaluation
```bash
python model_evaluation.py
```

#### 6. Visualization
```bash
python visualization.py
```

## Models Trained

The project trains and compares 6 different regression models:

1. **Linear Regression** - Simple baseline model
2. **Ridge Regression** - L2 regularized linear regression
3. **Lasso Regression** - L1 regularized linear regression
4. **Random Forest** - Ensemble of decision trees
5. **Gradient Boosting** - Sequential ensemble learning
6. **Support Vector Regression** - Kernel-based regression

## Key Features

### Data Preprocessing
- Handling missing values
- Removing duplicates
- Label encoding for categorical variables
- Feature scaling (standardization)

### Feature Engineering
- Car age calculation
- Annual mileage computation
- Feature interactions
- Automatic feature selection using statistical tests

### Model Evaluation
- Train-test split (80-20)
- Cross-validation (5-fold)
- Multiple evaluation metrics:
  - R² Score (Coefficient of Determination)
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)

### Visualizations
- Price distribution histogram
- Feature correlation heatmap
- Actual vs predicted scatter plots
- Residuals analysis
- Model comparison bar charts
- Feature importance plots (for tree-based models)

## Output Files

### Plots (in `plots/` directory)
- `price_distribution.png` - Distribution of car prices
- `feature_correlation.png` - Feature correlation heatmap
- `model_comparison.png` - Comparison of all models
- `actual_vs_predicted_*.png` - Actual vs predicted plots for each model
- `residuals_*.png` - Residuals analysis
- `feature_importance_*.png` - Feature importance (for tree-based models)

### Models (in `models/` directory)
- `best_model_*.pkl` - Saved best performing model

## Results Interpretation

### R² Score
- Ranges from 0 to 1 (higher is better)
- Indicates proportion of variance explained by the model
- R² = 0.85 means the model explains 85% of price variance

### RMSE and MAE
- Both measure prediction error in price units (millions)
- Lower values indicate better predictions
- RMSE penalizes larger errors more than MAE

### Model Comparison
The final report ranks all models by performance metrics, helping you choose the best model for deployment.

## Real-World Applications

1. **Price Valuation**: Estimate fair market price for used cars
2. **Inventory Management**: Optimize pricing strategy for dealers
3. **Insurance**: Calculate vehicle value for premium determination
4. **Investment Analysis**: Assess asset depreciation over time
5. **Market Analysis**: Understand pricing trends and factors

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **scipy**: Scientific computing

## Customization

### Modify Model Parameters
Edit `model_training.py` to adjust hyperparameters:
```python
# Example: Change Random Forest parameters
rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
```

### Change Feature Count
Edit `main.py` to select different number of features:
```python
X_selected, selected_features = select_best_features(X_engineered, y, k=15)  # Select top 15 features
```

### Adjust Train-Test Split
Edit `model_training.py`:
```python
X_train, X_test, y_train, y_test = split_data(X_selected, y, test_size=0.15)  # 85-15 split
```

## Performance Optimization

For better results:
1. Collect more data
2. Engineer domain-specific features
3. Tune hyperparameters
4. Experiment with ensemble methods
5. Handle outliers appropriately

## Troubleshooting

### Missing Dataset
Ensure `dataset/car data.csv` exists in the project directory.

### Module Import Errors
Run: `pip install -r requirements.txt`

### Memory Issues
For large datasets, reduce the number of cross-validation folds:
```python
cv_results = evaluate_with_cross_validation(X_train, y_train, models, cv=3)
```

## Future Enhancements

- [ ] Web interface for price predictions
- [ ] API deployment (Flask/FastAPI)
- [ ] Real-time model updates
- [ ] Advanced feature engineering
- [ ] Deep learning models (Neural Networks)
- [ ] Time-series analysis
- [ ] Interactive Jupyter notebooks

## Author

Created as part of the Task-CarPrice machine learning project.

## License

This project is open source and available for educational and commercial use.

## References

- Scikit-learn Documentation: https://scikit-learn.org/
- Pandas Documentation: https://pandas.pydata.org/
- Matplotlib Documentation: https://matplotlib.org/
- Machine Learning by Tom Mitchell

---

**Happy Predicting!** 🚗📈
