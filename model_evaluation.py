"""BMW Pricing Challenge - Model Evaluation Utilities

Comprehensive model evaluation tools including metrics calculation,
visualization, and comparison functionality.

Author: Abhinav Rana
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')

class ModelEvaluator:
    """Comprehensive model evaluation toolkit."""
    
    def __init__(self):
        """Initialize the evaluator with results storage."""
        self.results = {}
        
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """Evaluate a single model with comprehensive metrics.
        
        Args:
            model: Trained sklearn model
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            model_name (str): Name of the model
            
        Returns:
            dict: Dictionary containing all metrics
        """
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'Model': model_name,
            'Train_R2': r2_score(y_train, train_pred),
            'Test_R2': r2_score(y_test, test_pred),
            'Train_RMSE': np.sqrt(mean_squared_error(y_train, train_pred)),
            'Test_RMSE': np.sqrt(mean_squared_error(y_test, test_pred)),
            'Train_MAE': mean_absolute_error(y_train, train_pred),
            'Test_MAE': mean_absolute_error(y_test, test_pred),
            'Predictions': test_pred
        }
        
        # Calculate overfitting indicator
        metrics['Overfitting'] = metrics['Train_R2'] - metrics['Test_R2']
        
        # Store results
        self.results[model_name] = metrics
        
        return metrics
    
    def cross_validate_model(self, model, X, y, cv=5):
        """Perform cross-validation on a model.
        
        Args:
            model: Sklearn model (untrained)
            X: Features
            y: Target
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation scores
        """
        cv_scores = cross_val_score(model, X, y, cv=cv, 
                                    scoring='r2', n_jobs=-1)
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
    
    def get_results_dataframe(self):
        """Convert results to a formatted DataFrame.
        
        Returns:
            pd.DataFrame: Results in tabular format
        """
        results_list = []
        for model_name, metrics in self.results.items():
            row = {
                'Model': model_name,
                'R² Score': f"{metrics['Test_R2']:.4f}",
                'RMSE (€)': f"{metrics['Test_RMSE']:.2f}",
                'MAE (€)': f"{metrics['Test_MAE']:.2f}",
                'Overfitting': f"{metrics['Overfitting']:.4f}"
            }
            results_list.append(row)
        
        df = pd.DataFrame(results_list)
        df = df.sort_values('R² Score', ascending=False).reset_index(drop=True)
        return df
    
    def print_results(self):
        """Print formatted results table."""
        print("\n" + "="*90)
        print("🏆 MODEL EVALUATION RESULTS")
        print("="*90)
        
        df = self.get_results_dataframe()
        print(df.to_string(index=False))
        
        # Find best model
        best_model = max(self.results.items(), 
                        key=lambda x: x[1]['Test_R2'])
        
        print("\n" + "-"*90)
        print(f"⭐ Best Model: {best_model[0]}")
        print(f"   R² Score: {best_model[1]['Test_R2']:.4f}")
        print(f"   RMSE: €{best_model[1]['Test_RMSE']:.2f}")
        print(f"   MAE: €{best_model[1]['Test_MAE']:.2f}")
        print("="*90)
    
    def plot_model_comparison(self, save_path='model_comparison.png'):
        """Create comprehensive model comparison visualizations.
        
        Args:
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Prepare data
        models = list(self.results.keys())
        r2_scores = [self.results[m]['Test_R2'] for m in models]
        rmse_scores = [self.results[m]['Test_RMSE'] for m in models]
        
        # Plot 1: R² Scores
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars1 = axes[0].barh(models, r2_scores, color=colors, edgecolor='black')
        axes[0].set_xlabel('R² Score', fontsize=12, fontweight='bold')
        axes[0].set_title('Model Performance (R² Score)', fontsize=14, fontweight='bold')
        axes[0].set_xlim(0, 1)
        axes[0].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars1, r2_scores):
            axes[0].text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{score:.4f}', va='center', fontweight='bold')
        
        # Plot 2: RMSE
        bars2 = axes[1].barh(models, rmse_scores, color='salmon', edgecolor='black')
        axes[1].set_xlabel('RMSE (€)', fontsize=12, fontweight='bold')
        axes[1].set_title('Model Error (RMSE)', fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars2, rmse_scores):
            axes[1].text(score + 100, bar.get_y() + bar.get_height()/2, 
                        f'€{score:.0f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {save_path}")
        plt.close()
    
    def plot_predictions(self, y_test, model_name=None, save_path='predictions_plot.png'):
        """Plot actual vs predicted values for a specific model.
        
        Args:
            y_test: Actual test values
            model_name (str): Name of model to plot (default: best model)
            save_path (str): Path to save the plot
        """
        if model_name is None:
            # Use best model
            model_name = max(self.results.items(), 
                           key=lambda x: x[1]['Test_R2'])[0]
        
        predictions = self.results[model_name]['Predictions']
        r2 = self.results[model_name]['Test_R2']
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions, alpha=0.6, edgecolors='k', s=50)
        
        # Perfect prediction line
        min_val = min(y_test.min(), predictions.min())
        max_val = max(y_test.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 
                'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Price (€)', fontsize=12, fontweight='bold')
        plt.ylabel('Predicted Price (€)', fontsize=12, fontweight='bold')
        plt.title(f'{model_name} - Actual vs Predicted\nR² = {r2:.4f}', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {save_path}")
        plt.close()
    
    def plot_residuals(self, y_test, model_name=None, save_path='residuals_plot.png'):
        """Plot residuals for model diagnostics.
        
        Args:
            y_test: Actual test values
            model_name (str): Name of model to plot (default: best model)
            save_path (str): Path to save the plot
        """
        if model_name is None:
            model_name = max(self.results.items(), 
                           key=lambda x: x[1]['Test_R2'])[0]
        
        predictions = self.results[model_name]['Predictions']
        residuals = y_test - predictions
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Residuals scatter
        axes[0].scatter(predictions, residuals, alpha=0.6, edgecolors='k')
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Price (€)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Residuals (€)', fontsize=12, fontweight='bold')
        axes[0].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals distribution
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residuals (€)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Diagnostic Plots', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {save_path}")
        plt.close()


def evaluate_models(models_dict, X_train, X_test, y_train, y_test, plot=True):
    """Evaluate multiple models and generate comprehensive reports.
    
    Args:
        models_dict (dict): Dictionary of {'model_name': trained_model}
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        plot (bool): Whether to generate visualizations
        
    Returns:
        ModelEvaluator: Evaluator object with all results
    """
    print("\n" + "="*80)
    print("📊 EVALUATING MODELS")
    print("="*80)
    
    evaluator = ModelEvaluator()
    
    for model_name, model in models_dict.items():
        print(f"\n⏳ Evaluating {model_name}...")
        evaluator.evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
        print(f"   ✓ Complete")
    
    # Print results
    evaluator.print_results()
    
    # Generate plots
    if plot:
        print("\n📊 Generating visualizations...")
        evaluator.plot_model_comparison()
        evaluator.plot_predictions(y_test)
        evaluator.plot_residuals(y_test)
    
    return evaluator


if __name__ == "__main__":
    print("\n📊 BMW Model Evaluation Module")
    print("This module provides utilities for evaluating ML models.")
    print("\nImport this module to use:")
    print("  from model_evaluation import ModelEvaluator, evaluate_models")
    print("\nExample:")
    print("  evaluator = evaluate_models(models_dict, X_train, X_test, y_train, y_test)")
