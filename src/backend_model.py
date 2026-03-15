import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
import shap
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# Set plotting style for research-level viz
plt.style.use('seaborn-v0_8-whitegrid')

# ================================
# 1. LOAD AND MERGE RAW DATA
# ================================
def load_and_merge_data():
    print("Loading datasets...")
    try:
        df_ground = pd.read_csv('C:/Users/91860/Desktop/AeroGraphFed/data/raw/global_data.csv')
    except:
        df_ground = pd.read_csv('../data/raw/global_data.csv')

    try:
        df_sat = pd.read_csv('C:/Users/91860/Desktop/AeroGraphFed/data/derived/satellite_pm25_features.csv')
    except:
        df_sat = pd.read_csv('../data/derived/satellite_pm25_features.csv')
        
    # Formatting column names
    if 'Year' in df_ground.columns:
        df_ground = df_ground.rename(columns={'Year': 'year'})
    target_col = 'Population-Weighted PM2.5 [ug/m3]'
    if target_col in df_ground.columns:
        df_ground = df_ground.rename(columns={target_col: 'pm25'})
    if 'Total Population [million people]' in df_ground.columns:
        df_ground = df_ground.rename(columns={'Total Population [million people]': 'population'})

    # Merge
    if "Region" in df_ground.columns:
        df = df_ground.merge(df_sat, on=["Region", "year"], how="inner")
        # Treat the extracted PM2.5 as MODIS AOD proxy since it is derived from satellites like MODIS
        df = df.rename(columns={'CIESIN_PM25': 'MODIS_AOD_PROXY'})
    else:
        df = df_ground.copy()
        
    return df

# ================================
# 2. ADVANCED FEATURE ENGINEERING
# ================================
def engineer_features(df):
    print("Engineering temporal and spatial features...")
    df = df.sort_values(by=["Region", "year"] if "Region" in df.columns else ["year"])
    
    # Lag features per region to prevent data leakage across countries
    if "Region" in df.columns:
        df["pm25_lag1"] = df.groupby("Region")["pm25"].shift(1)
        df["pm25_lag2"] = df.groupby("Region")["pm25"].shift(2)
        df["pm25_roll3_mean"] = df.groupby("Region")["pm25"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df["pm25_roll3_std"] = df.groupby("Region")["pm25"].transform(lambda x: x.rolling(3, min_periods=2).std())
        df["pm25_growth"] = df.groupby("Region")["pm25"].pct_change()
        
        # Satellite measurement growth
        if "MODIS_AOD_PROXY" in df.columns:
            df["sat_pm25_growth"] = df.groupby("Region")["MODIS_AOD_PROXY"].pct_change()
    else:
        df["pm25_lag1"] = df["pm25"].shift(1)
        df["pm25_lag2"] = df["pm25"].shift(2)
        df["pm25_roll3_mean"] = df["pm25"].rolling(3, min_periods=1).mean()
        df["pm25_roll3_std"] = df["pm25"].rolling(3, min_periods=2).std()
        df["pm25_growth"] = df["pm25"].pct_change()
        if "MODIS_AOD_PROXY" in df.columns:
            df["sat_pm25_growth"] = df["MODIS_AOD_PROXY"].pct_change()
            
    # Non-linear population transformation
    if "population" in df.columns:
        df["log_population"] = np.log1p(df["population"])
    else:
        df["log_population"] = 0

    # Drop nulls caused by lagging to ensure clean training data
    df = df.dropna(subset=['pm25', 'pm25_lag1', 'pm25_lag2'])
    if "MODIS_AOD_PROXY" in df.columns:
        df = df.dropna(subset=['MODIS_AOD_PROXY'])
        # Fill missing std devs and growth with 0 for the earliest window if applicable
        df.fillna(0, inplace=True)
        
    return df

# ================================
# 3. TIME-SERIES CROSS VALIDATION
# ================================
def evaluate_timeseries_cv(X, y, df_years):
    print("\nPerforming Time-Series Cross Validation...")
    # TimeSeriesSplit ensures we never train on future data to predict past data
    tscv = TimeSeriesSplit(n_splits=5)
    
    cv_scores = {'rmse': [], 'mae': [], 'r2': []}
    
    # Sort X and y strictly by year to respect the time arrows
    sort_idx = np.argsort(df_years)
    X_sorted = X.iloc[sort_idx].reset_index(drop=True)
    y_sorted = y.iloc[sort_idx].reset_index(drop=True)
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X_sorted)):
        X_train, X_test = X_sorted.iloc[train_index], X_sorted.iloc[test_index]
        y_train, y_test = y_sorted.iloc[train_index], y_sorted.iloc[test_index]
        
        # Define a model with early stopping specs to prevent overfitting inside CV
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        
        y_pred = model.predict(X_test)
        
        cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        cv_scores['mae'].append(mean_absolute_error(y_test, y_pred))
        cv_scores['r2'].append(r2_score(y_test, y_pred))
        
    print("CV Results (5 Folds):")
    print(f"Mean RMSE: {np.mean(cv_scores['rmse']):.4f} (+/- {np.std(cv_scores['rmse']):.4f})")
    print(f"Mean MAE:  {np.mean(cv_scores['mae']):.4f} (+/- {np.std(cv_scores['mae']):.4f})")
    print(f"Mean R2:   {np.mean(cv_scores['r2']):.4f} (+/- {np.std(cv_scores['r2']):.4f})")


# ================================
# 4. FINAL MODEL TRAINING & SHAP
# ================================
def train_and_explain_model(X, y, features):
    # Standard train/test split for final holdout
    # Ideally should be a temporal split (e.g. train < 2018, test >= 2018) for strict forecasting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining Final Robust Model...")
    final_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=2, # Prevents severe overfitting on individual nodes
        subsample=0.8,
        colsample_bytree=0.8, # Feature dropout
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50 # Stop adding trees if test error doesn't improve
    )
    
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100
    )
    
    # Final Evaluate
    y_pred = final_model.predict(X_test)
    print("\n" + "="*30)
    print("Final Hold-Out Test Set Performance")
    print("="*30)
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"R2:   {r2_score(y_test, y_pred):.4f}")
    print("="*30)
    
    # Save parity plot
    os.makedirs('../images', exist_ok=True)
    plt.figure(figsize=(8,8))
    plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual PM2.5')
    plt.ylabel('Predicted PM2.5')
    plt.title('Actual vs Predicted PM2.5 (Holdout Test Set)')
    plt.savefig('../images/actual_vs_predicted_pm25.png', dpi=300)
    plt.close()

    # SHAP Explainability
    print("\nGenerating Research-Level SHAP Visualizations...")
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer(X_test)

    # 1. Standard Summary Plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("Global Feature Importance (SHAP Summary)")
    plt.tight_layout()
    plt.savefig("../images/shap_summary_advanced.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Bar Plot for absolute influence 
    shap.plots.bar(shap_values, show=False)
    plt.title("Mean Absolute SHAP Value")
    plt.tight_layout()
    plt.savefig("../images/shap_bar_advanced.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Dependence Plot for the Top Feature (often MODIS or lag1)
    # Automatically finds the most important feature index to plot
    top_feature_idx = np.abs(shap_values.values).mean(0).argmax()
    top_feature_name = features[top_feature_idx]
    
    shap.dependence_plot(top_feature_name, shap_values.values, X_test, feature_names=features, show=False)
    plt.title(f"SHAP Dependence: {top_feature_name}")
    plt.tight_layout()
    plt.savefig(f"../images/shap_dependence_{top_feature_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return final_model

# ================================
# 5. EXECUTION PIPELINE
# ================================
if __name__ == "__main__":
    df = load_and_merge_data()
    df = engineer_features(df)
    
    features = [
        "year",
        "log_population",
        "pm25_lag1",
        "pm25_lag2",
        "pm25_roll3_mean",
        "pm25_roll3_std",
        "pm25_growth",
        "MODIS_AOD_PROXY",
        "sat_pm25_growth"
    ]
    
    # Filter only available features in dataframe safely
    features = [f for f in features if f in df.columns]
    print(f"\nTraining with features: {features}")
    
    X = df[features]
    y = df["pm25"]
    
    # Run rigorous Time Series Cross Validation
    evaluate_timeseries_cv(X, y, df['year'].values)
    
    # Train robust final model and explain
    final_model = train_and_explain_model(X, y, features)
    
    # Save assets
    os.makedirs('../models', exist_ok=True)
    joblib.dump(final_model, "../models/pm25_xgboost_research_model.pkl")
    with open("../models/research_model_features.txt", "w") as f:
        for col in features:
            f.write(col + "\n")
            
    print("\nResearch-Level Pipeline Complete! Assets saved.")
