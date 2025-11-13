#!/usr/bin/env python3
"""
BMW Pricing Challenge - Complete ML Analysis Pipeline

Author: Abhinav Rana
Date: November 2025

One-command execution: python run_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

print("="*80)
print("BMW PRICING CHALLENGE - MACHINE LEARNING PIPELINE")
print("Author: Abhinav Rana")
print("="*80)
print()

# Load Data
print("✅ Step 1: Loading BMW Dataset...")
df = pd.read_csv('bmw_pricing_challenge.csv')
print(f"   Dataset shape: {df.shape}")
print(f"   BMW records: {len(df):,}")

# Parse dates
df['registration_date'] = pd.to_datetime(df['registration_date'])
current_year = datetime.now().year
df['Age'] = current_year - df['registration_date'].dt.year

# Feature Engineering
print("\n✅ Step 2: Feature Engineering...")
df['Mileage_per_Year'] = df['mileage'] / (df['Age'] + 1)
df['Price_per_HP'] = df['sold_at'] / (df['engine_power'] + 1)

# Count boolean features
feature_cols = [col for col in df.columns if col.startswith('feature_')]
if feature_cols:
    df['Feature_Count'] = df[feature_cols].sum(axis=1)
    print(f"   Engineered features: Age, Mileage_per_Year, Price_per_HP, Feature_Count")

# Encode categorical variables
print("\n✅ Step 3: Encoding Categorical Variables...")
le_dict = {}
for col in ['model_key', 'fuel', 'paint_color', 'car_type']:
    le_dict[col] = LabelEncoder()
    df[f'{col}_encoded'] = le_dict[col].fit_transform(df[col].astype(str))
    print(f"   {col}: {df[col].nunique()} unique values")

# Select features for modeling
feature_list = ['Age', 'mileage', 'engine_power', 'Mileage_per_Year',
                'model_key_encoded', 'fuel_encoded', 'paint_color_encoded', 
                'car_type_encoded']

if 'Feature_Count' in df.columns:
    feature_list.append('Feature_Count')

X = df[feature_list]
y = df['sold_at']

# Split data
print("\n✅ Step 4: Train-Test Split (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"   Training set: {X_train_scaled.shape[0]} samples")
print(f"   Test set: {X_test_scaled.shape[0]} samples")

# Train models
print("\n✅ Step 5: Training ML Models...")
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

if XGBOOST_AVAILABLE:
    models['XGBoost'] = XGBRegressor(random_state=42, verbosity=0)

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'predictions': y_pred}
    print(f"   ✓ {name:20s} - R²: {r2:.4f}")

# Print Results
print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON")
print("="*80)

results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'R² Score': [results[m]['R2'] for m in results.keys()],
    'RMSE (€)': [results[m]['RMSE'] for m in results.keys()],
    'MAE (€)': [results[m]['MAE'] for m in results.keys()]
}).sort_values('R² Score', ascending=False)

print("\n" + results_df.to_string(index=False))

best_model = results_df.iloc[0]['Model']
best_r2 = results_df.iloc[0]['R² Score']

print(f"\n🏆 Best Model: {best_model}")
print(f"   R² Score: {best_r2:.4f}")
print(f"   Explains {best_r2*100:.2f}% of BMW price variance")

# Visualizations
print("\n✅ Step 6: Generating Visualizations...")

# 1. Model Comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(results_df['Model'], results_df['R² Score'], color='skyblue', edgecolor='black')
plt.xlabel('R² Score', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

plt.subplot(1, 2, 2)
plt.barh(results_df['Model'], results_df['RMSE (€)'], color='salmon', edgecolor='black')
plt.xlabel('RMSE (€)', fontsize=12)
plt.title('Model Error Comparison', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: model_comparison.png")

# 2. Actual vs Predicted (Best Model)
plt.figure(figsize=(10, 6))
y_pred_best = results[best_model]['predictions']
plt.scatter(y_test, y_pred_best, alpha=0.6, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price (€)', fontsize=12)
plt.ylabel('Predicted Price (€)', fontsize=12)
plt.title(f'{best_model} - Actual vs Predicted\nR² = {best_r2:.4f}', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: actual_vs_predicted.png")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated Files:")
print("  - model_comparison.png")
print("  - actual_vs_predicted.png")
print("\n🚀 Your BMW Pricing Challenge project is ready for recruiters!")
print("="*80)
