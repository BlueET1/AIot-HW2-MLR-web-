"""
analysis.py - HW2: Multiple Linear Regression (CRISP-DM Workflow)
Dataset: Life Expectancy (WHO)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_and_clean_data(file_path="data/Life_Expectancy_Data.csv"):
    df = pd.read_csv(file_path)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Drop rows missing the target variable
    df = df.dropna(subset=['Life expectancy']).reset_index(drop=True)
    
    # Target and predictor separation
    target_col = 'Life expectancy'
    y = df[target_col]
    
    # Exclude non-predictive metadata (Country)
    X = df.drop(columns=['Country', target_col])
    
    # Encode categorical variable 'Status' (1 = Developed, 0 = Developing)
    if 'Status' in X.columns:
        X['Status'] = (X['Status'] == 'Developed').astype(int)
        
    # Impute missing values in features with median
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
            
    return df, X, y

def compute_vif(X):
    X_const = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    return vif_data.sort_values(by="VIF", ascending=False)

def run_crisp_dm_pipeline():
    print("="*60)
    print("CRISP-DM Step 1: Business Understanding")
    print("Objective: Predict global life expectancy using public health & economic factors.")
    print("="*60)
    
    print("\nCRISP-DM Step 2: Data Understanding")
    df, X, y = load_and_clean_data()
    print(f"Sample size: {len(df)} rows, {X.shape[1]} features.")
    print("Features list:", list(X.columns))
    
    print("\nCRISP-DM Step 3: Data Preparation")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Feature Selection: Compare full model vs Lasso selected vs RFE selected
    print("\n" + "="*60)
    print("Feature Selection Analysis")
    print("="*60)
    
    # 1. Lasso Selection
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lasso_cv = LassoCV(cv=5, random_state=42).fit(X_train_scaled, y_train)
    lasso_coefs = pd.Series(lasso_cv.coef_, index=X.columns)
    selected_by_lasso = lasso_coefs[lasso_coefs.abs() > 0.05].index.tolist()
    print(f"Lasso selected {len(selected_by_lasso)} / {X.shape[1]} features:")
    print(selected_by_lasso)
    
    # 2. RFE (Select top 14 features)
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=14)
    rfe.fit(X_train, y_train)
    selected_by_rfe = X.columns[rfe.support_].tolist()
    print(f"\nRFE selected top 14 features:")
    print(selected_by_rfe)
    
    print("\nCRISP-DM Step 4 & 5: Modeling and Evaluation (OLS with Confidence & Prediction Intervals)")
    # Using statsmodels OLS on selected features (or full) for statistical inference
    X_train_ols = sm.add_constant(X_train[selected_by_rfe])
    X_test_ols = sm.add_constant(X_test[selected_by_rfe])
    
    ols_model = sm.OLS(y_train, X_train_ols).fit()
    print(ols_model.summary())
    
    # Predictions with Confidence and Prediction Intervals
    pred_obj = ols_model.get_prediction(X_test_ols)
    pred_frame = pred_obj.summary_frame(alpha=0.05)
    
    y_pred = pred_frame['mean']
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("Model Evaluation Metrics on Test Set:")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print("="*60)
    
    # Save a sample plot
    os.makedirs("output", exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    # Sort for clear visualization
    sorted_indices = np.argsort(y_test.values)
    y_test_sorted = y_test.values[sorted_indices]
    y_pred_sorted = y_pred.values[sorted_indices]
    ci_lower = pred_frame['mean_ci_lower'].values[sorted_indices]
    ci_upper = pred_frame['mean_ci_upper'].values[sorted_indices]
    pi_lower = pred_frame['obs_ci_lower'].values[sorted_indices]
    pi_upper = pred_frame['obs_ci_upper'].values[sorted_indices]
    
    x_axis = np.arange(len(y_test_sorted))
    
    # Take a representative subset for plotting (e.g. 60 points) so intervals are clearly visible
    sample_idx = np.linspace(0, len(y_test_sorted)-1, 60, dtype=int)
    
    plt.plot(x_axis[sample_idx], y_test_sorted[sample_idx], 'ko', markersize=4, label='Actual Values')
    plt.plot(x_axis[sample_idx], y_pred_sorted[sample_idx], 'r-', linewidth=1.5, label='Predicted Mean (OLS)')
    plt.fill_between(x_axis[sample_idx], ci_lower[sample_idx], ci_upper[sample_idx], color='red', alpha=0.3, label='95% Confidence Interval (CI)')
    plt.fill_between(x_axis[sample_idx], pi_lower[sample_idx], pi_upper[sample_idx], color='orange', alpha=0.15, label='95% Prediction Interval (PI)')
    
    plt.title('Multiple Linear Regression: Actual vs Predicted with 95% CI and PI', fontsize=12)
    plt.xlabel('Test Samples (Sorted by Actual Value)', fontsize=10)
    plt.ylabel('Life Expectancy (Years)', fontsize=10)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('output/prediction_interval_plot.png', dpi=300)
    print("Prediction plot saved to output/prediction_interval_plot.png")

if __name__ == "__main__":
    run_crisp_dm_pipeline()

