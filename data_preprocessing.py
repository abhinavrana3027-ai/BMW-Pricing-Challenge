"""BMW Pricing Challenge - Data Preprocessing Utilities

This module provides reusable data preprocessing functions for the BMW pricing analysis.
Includes feature engineering, encoding, scaling, and data cleaning utilities.

Author: Abhinav Rana
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class BMWDataPreprocessor:
    """Handles all preprocessing operations for BMW pricing data."""
    
    def __init__(self, data):
        """Initialize with raw BMW pricing data.
        
        Args:
            data (pd.DataFrame): Raw BMW pricing dataset
        """
        self.data = data.copy()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def clean_data(self):
        """Remove duplicates and handle missing values.
        
        Returns:
            pd.DataFrame: Cleaned data
        """
        print("\n🧹 Cleaning data...")
        initial_rows = len(self.data)
        
        # Remove duplicates
        self.data = self.data.drop_duplicates()
        
        # Handle missing values
        self.data = self.data.dropna()
        
        removed_rows = initial_rows - len(self.data)
        print(f"   ✓ Removed {removed_rows} rows (duplicates/missing values)")
        print(f"   ✓ Final dataset: {len(self.data)} rows")
        
        return self.data
    
    def engineer_features(self):
        """Create engineered features from existing columns.
        
        Returns:
            pd.DataFrame: Data with new engineered features
        """
        print("\n⚙️  Engineering features...")
        
        # Age calculation
        if 'model_year' in self.data.columns:
            current_year = 2024
            self.data['age'] = current_year - self.data['model_year']
            print(f"   ✓ Created 'age' feature (range: {self.data['age'].min()}-{self.data['age'].max()} years)")
        
        # Mileage per year
        if 'mileage' in self.data.columns and 'age' in self.data.columns:
            self.data['mileage_per_year'] = self.data['mileage'] / (self.data['age'] + 1)
            print(f"   ✓ Created 'mileage_per_year' feature")
        
        # Price per horsepower (efficiency indicator)
        if 'engine_power' in self.data.columns and 'price' in self.data.columns:
            self.data['price_per_hp'] = self.data['price'] / (self.data['engine_power'] + 1)
            print(f"   ✓ Created 'price_per_hp' feature")
        
        # Feature richness score
        feature_cols = ['automatic_transmission', 'fuel', 'paint_color', 'car_type']
        existing_features = [col for col in feature_cols if col in self.data.columns]
        if existing_features:
            self.data['feature_count'] = self.data[existing_features].notna().sum(axis=1)
            print(f"   ✓ Created 'feature_count' feature")
        
        return self.data
    
    def encode_categorical(self, columns):
        """Encode categorical variables using Label Encoding.
        
        Args:
            columns (list): List of column names to encode
            
        Returns:
            pd.DataFrame: Data with encoded categorical variables
        """
        print("\n🔢 Encoding categorical variables...")
        
        for col in columns:
            if col in self.data.columns:
                self.label_encoders[col] = LabelEncoder()
                self.data[col] = self.label_encoders[col].fit_transform(self.data[col].astype(str))
                n_categories = len(self.label_encoders[col].classes_)
                print(f"   ✓ Encoded '{col}' ({n_categories} categories)")
        
        return self.data
    
    def prepare_features(self, target_column='price', test_size=0.2, random_state=42):
        """Prepare final feature set and split into train/test.
        
        Args:
            target_column (str): Name of target variable
            test_size (float): Proportion of test set
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, scaler)
        """
        print("\n📊 Preparing features and splitting data...")
        
        # Separate features and target
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
        
        print(f"   ✓ Training set: {len(X_train)} samples")
        print(f"   ✓ Test set: {len(X_test)} samples")
        print(f"   ✓ Features: {X.shape[1]} columns")
        print(f"   ✓ Features scaled using StandardScaler")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, self.scaler
    
    def get_feature_info(self):
        """Display information about processed features.
        
        Returns:
            pd.DataFrame: Feature information summary
        """
        info_dict = {
            'Feature': self.data.columns.tolist(),
            'Type': [self.data[col].dtype for col in self.data.columns],
            'Missing': [self.data[col].isnull().sum() for col in self.data.columns],
            'Unique': [self.data[col].nunique() for col in self.data.columns]
        }
        
        return pd.DataFrame(info_dict)


def preprocess_bmw_data(data, target='price', categorical_cols=None):
    """Complete preprocessing pipeline for BMW pricing data.
    
    Args:
        data (pd.DataFrame): Raw BMW pricing data
        target (str): Target variable name
        categorical_cols (list): List of categorical columns to encode
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, preprocessor)
    """
    if categorical_cols is None:
        categorical_cols = ['model_key', 'fuel', 'paint_color', 'car_type']
    
    print("\n" + "="*80)
    print("🚗 BMW PRICING CHALLENGE - DATA PREPROCESSING PIPELINE")
    print("="*80)
    
    # Initialize preprocessor
    preprocessor = BMWDataPreprocessor(data)
    
    # Step 1: Clean data
    preprocessor.clean_data()
    
    # Step 2: Engineer features
    preprocessor.engineer_features()
    
    # Step 3: Encode categorical variables
    preprocessor.encode_categorical(categorical_cols)
    
    # Step 4: Prepare features and split
    X_train, X_test, y_train, y_test, scaler = preprocessor.prepare_features(target)
    
    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE")
    print("="*80)
    
    return X_train, X_test, y_train, y_test, scaler, preprocessor


if __name__ == "__main__":
    # Example usage
    print("\n📦 BMW Data Preprocessing Module")
    print("This module provides utilities for preprocessing BMW pricing data.")
    print("\nImport this module to use:")
    print("  from data_preprocessing import preprocess_bmw_data, BMWDataPreprocessor")
    print("\nExample:")
    print("  X_train, X_test, y_train, y_test, scaler, preprocessor = preprocess_bmw_data(df)")
