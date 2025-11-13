"""BMW Pricing Challenge - Prediction Module

Production-ready inference module for predicting BMW car prices.
Provides both single predictions and batch predictions with confidence intervals.

Author: Abhinav Rana
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BMWPricePredictor:
    """Production predictor for BMW car prices."""
    
    def __init__(self, model_path=None, scaler_path=None):
        """Initialize the predictor with saved model and scaler.
        
        Args:
            model_path (str): Path to saved model pickle file
            scaler_path (str): Path to saved scaler pickle file
        """
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
        if scaler_path:
            self.load_scaler(scaler_path)
    
    def load_model(self, model_path):
        """Load a trained model from disk.
        
        Args:
            model_path (str): Path to model pickle file
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_scaler(self, scaler_path):
        """Load a fitted scaler from disk.
        
        Args:
            scaler_path (str): Path to scaler pickle file
        """
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ Scaler loaded successfully from {scaler_path}")
        except Exception as e:
            print(f"❌ Error loading scaler: {e}")
            raise
    
    def save_model(self, model, save_path):
        """Save a trained model to disk.
        
        Args:
            model: Trained sklearn model
            save_path (str): Path to save the model
        """
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ Model saved successfully to {save_path}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            raise
    
    def save_scaler(self, scaler, save_path):
        """Save a fitted scaler to disk.
        
        Args:
            scaler: Fitted sklearn scaler
            save_path (str): Path to save the scaler
        """
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"✅ Scaler saved successfully to {save_path}")
        except Exception as e:
            print(f"❌ Error saving scaler: {e}")
            raise
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction.
        
        Args:
            input_data (dict or pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Preprocessed features ready for prediction
        """
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Apply feature engineering (same as training)
        if 'model_year' in input_data.columns:
            current_year = 2024
            input_data['age'] = current_year - input_data['model_year']
        
        if 'mileage' in input_data.columns and 'age' in input_data.columns:
            input_data['mileage_per_year'] = input_data['mileage'] / (input_data['age'] + 1)
        
        if 'engine_power' in input_data.columns:
            # Note: price_per_hp cannot be calculated without actual price
            pass
        
        # Scale features if scaler is available
        if self.scaler:
            input_scaled = self.scaler.transform(input_data)
            input_data = pd.DataFrame(input_scaled, columns=input_data.columns)
        
        return input_data
    
    def predict(self, input_data, return_confidence=False):
        """Make a single prediction.
        
        Args:
            input_data (dict or pd.DataFrame): Input features
            return_confidence (bool): Whether to return confidence interval
            
        Returns:
            float or dict: Predicted price (and confidence if requested)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess input
        X = self.preprocess_input(input_data)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        if return_confidence:
            # Estimate confidence interval (±10% for regression)
            confidence_margin = prediction * 0.10
            return {
                'predicted_price': float(prediction),
                'confidence_lower': float(prediction - confidence_margin),
                'confidence_upper': float(prediction + confidence_margin),
                'currency': 'EUR'
            }
        
        return float(prediction)
    
    def predict_batch(self, input_dataframe):
        """Make batch predictions.
        
        Args:
            input_dataframe (pd.DataFrame): DataFrame with multiple input samples
            
        Returns:
            np.array: Array of predicted prices
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess input
        X = self.preprocess_input(input_dataframe)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    def predict_with_features(self, model_year, mileage, engine_power, 
                            fuel='Diesel', transmission='Automatic',
                            paint_color='Black', car_type='Sedan'):
        """Make a prediction with individual feature values.
        
        Args:
            model_year (int): Year of manufacture
            mileage (int): Mileage in kilometers
            engine_power (int): Engine power in HP
            fuel (str): Fuel type
            transmission (str): Transmission type
            paint_color (str): Paint color
            car_type (str): Car body type
            
        Returns:
            dict: Prediction with details
        """
        input_data = {
            'model_year': model_year,
            'mileage': mileage,
            'engine_power': engine_power,
            'fuel': fuel,
            'transmission': transmission,
            'paint_color': paint_color,
            'car_type': car_type
        }
        
        prediction = self.predict(input_data, return_confidence=True)
        
        # Add input details to result
        result = {
            'input': input_data,
            'prediction': prediction
        }
        
        return result


def predict_from_csv(csv_path, model_path, scaler_path, output_path=None):
    """Make predictions on a CSV file.
    
    Args:
        csv_path (str): Path to input CSV file
        model_path (str): Path to saved model
        scaler_path (str): Path to saved scaler
        output_path (str): Path to save predictions (optional)
        
    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    print("\n" + "="*80)
    print("🔮 BMW PRICE PREDICTION SERVICE")
    print("="*80)
    
    # Load data
    print(f"\n📊 Loading data from {csv_path}...")
    data = pd.read_csv(csv_path)
    print(f"   ✓ Loaded {len(data)} samples")
    
    # Initialize predictor
    predictor = BMWPricePredictor(model_path, scaler_path)
    
    # Make predictions
    print("\n🔮 Making predictions...")
    predictions = predictor.predict_batch(data)
    
    # Add predictions to dataframe
    data['predicted_price'] = predictions
    
    print(f"   ✓ Predictions complete")
    print(f"\n📊 Prediction Statistics:")
    print(f"   Mean Price: €{predictions.mean():.2f}")
    print(f"   Median Price: €{np.median(predictions):.2f}")
    print(f"   Min Price: €{predictions.min():.2f}")
    print(f"   Max Price: €{predictions.max():.2f}")
    
    # Save if output path provided
    if output_path:
        data.to_csv(output_path, index=False)
        print(f"\n✅ Predictions saved to {output_path}")
    
    print("\n" + "="*80)
    
    return data


if __name__ == "__main__":
    print("\n🔮 BMW Price Prediction Module")
    print("\nThis module provides production-ready inference for BMW pricing.")
    print("\nExample Usage:")
    print("\n1. Single Prediction:")
    print("   predictor = BMWPricePredictor('model.pkl', 'scaler.pkl')")
    print("   price = predictor.predict({'model_year': 2020, 'mileage': 50000, ...})")
    print("\n2. Batch Prediction:")
    print("   df = predict_from_csv('input.csv', 'model.pkl', 'scaler.pkl', 'output.csv')")
    print("\n3. Feature-based Prediction:")
    print("   result = predictor.predict_with_features(")
    print("       model_year=2020, mileage=50000, engine_power=200)")
