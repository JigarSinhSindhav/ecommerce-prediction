"""
Machine Learning Models for E-commerce Sales Prediction
Implements Random Forest and XGBoost models with comprehensive evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SalesPredictionModels:
    def __init__(self):
        self.models = {}
        self.model_metrics = {}
        self.feature_importance = {}
        
    def train_random_forest(self, X_train, y_train, **kwargs):
        """Train Random Forest model"""
        print("Training Random Forest model...")
        
        # Default parameters
        rf_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        rf_params.update(kwargs)
        
        # Train model
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X_train, y_train)
        
        # Store model
        self.models['random_forest'] = rf_model
        
        # Store feature importance
        self.feature_importance['random_forest'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Random Forest training completed!")
        return rf_model
    
    def train_xgboost(self, X_train, y_train, **kwargs):
        """Train XGBoost model"""
        print("Training XGBoost model...")
        
        # Default parameters
        xgb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
        xgb_params.update(kwargs)
        
        # Train model
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train)
        
        # Store model
        self.models['xgboost'] = xgb_model
        
        # Store feature importance
        self.feature_importance['xgboost'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("XGBoost training completed!")
        return xgb_model
    
    def evaluate_model(self, model_name, X_test, y_test):
        """Evaluate a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate accuracy as 1 - (MAE / mean(actual))
        accuracy = 1 - (mae / y_test.mean())
        accuracy = max(0, accuracy)  # Ensure non-negative
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Accuracy': accuracy * 100  # Convert to percentage
        }
        
        self.model_metrics[model_name] = metrics
        
        print(f"\n{model_name.upper()} Model Evaluation:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        
        return metrics, y_pred
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("Evaluating all models...")
        
        predictions = {}
        
        for model_name in self.models.keys():
            metrics, y_pred = self.evaluate_model(model_name, X_test, y_test)
            predictions[model_name] = y_pred
        
        return predictions
    
    def plot_feature_importance(self, model_name, top_n=15):
        """Plot feature importance for a specific model"""
        if model_name not in self.feature_importance:
            print(f"Feature importance not available for {model_name}")
            return
        
        importance_df = self.feature_importance[model_name].head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, y='feature', x='importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name.upper()}')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        
        # Save plot
        os.makedirs('../models/plots', exist_ok=True)
        plt.savefig(f'../models/plots/{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions_comparison(self, X_test, y_test, sample_size=1000):
        """Plot actual vs predicted values for all models"""
        predictions = self.evaluate_all_models(X_test, y_test)
        
        # Sample data for better visualization
        if len(y_test) > sample_size:
            indices = np.random.choice(len(y_test), sample_size, replace=False)
            y_test_sample = y_test.iloc[indices]
        else:
            y_test_sample = y_test
            indices = y_test.index
        
        n_models = len(predictions)
        fig, axes = plt.subplots(1, n_models, figsize=(15, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            y_pred_sample = y_pred[indices] if len(y_test) > sample_size else y_pred
            
            axes[i].scatter(y_test_sample, y_pred_sample, alpha=0.6)
            axes[i].plot([y_test_sample.min(), y_test_sample.max()], 
                        [y_test_sample.min(), y_test_sample.max()], 'r--', lw=2)
            axes[i].set_xlabel('Actual Sales')
            axes[i].set_ylabel('Predicted Sales')
            axes[i].set_title(f'{model_name.upper()}\nR² = {self.model_metrics[model_name]["R2"]:.3f}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs('../models/plots', exist_ok=True)
        plt.savefig('../models/plots/predictions_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_residuals(self, X_test, y_test, sample_size=1000):
        """Plot residuals for all models"""
        predictions = {}
        for model_name in self.models.keys():
            _, y_pred = self.evaluate_model(model_name, X_test, y_test)
            predictions[model_name] = y_pred
        
        # Sample data for better visualization
        if len(y_test) > sample_size:
            indices = np.random.choice(len(y_test), sample_size, replace=False)
            y_test_sample = y_test.iloc[indices]
        else:
            y_test_sample = y_test
            indices = y_test.index
        
        n_models = len(predictions)
        fig, axes = plt.subplots(1, n_models, figsize=(15, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            y_pred_sample = y_pred[indices] if len(y_test) > sample_size else y_pred
            residuals = y_test_sample - y_pred_sample
            
            axes[i].scatter(y_pred_sample, residuals, alpha=0.6)
            axes[i].axhline(y=0, color='r', linestyle='--')
            axes[i].set_xlabel('Predicted Sales')
            axes[i].set_ylabel('Residuals')
            axes[i].set_title(f'{model_name.upper()} Residuals')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../models/plots/residuals_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self):
        """Create a comparison table of all model metrics"""
        if not self.model_metrics:
            print("No models have been evaluated yet.")
            return None
        
        comparison_df = pd.DataFrame(self.model_metrics).T
        comparison_df = comparison_df.round(4)
        
        print("\nModel Comparison:")
        print(comparison_df)
        
        # Plot comparison
        metrics_to_plot = ['R2', 'Accuracy']
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(12, 5))
        
        for i, metric in enumerate(metrics_to_plot):
            comparison_df[metric].plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'{metric} Comparison')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('../models/plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df
    
    def save_models(self, model_dir='../models'):
        """Save trained models to disk"""
        os.makedirs(model_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(model_dir, f'{model_name}_model.joblib')
            joblib.dump(model, model_path)
            print(f"Saved {model_name} model to {model_path}")
        
        # Save metrics
        metrics_path = os.path.join(model_dir, 'model_metrics.joblib')
        joblib.dump(self.model_metrics, metrics_path)
        
        # Save feature importance
        importance_path = os.path.join(model_dir, 'feature_importance.joblib')
        joblib.dump(self.feature_importance, importance_path)
        
        print(f"Saved model metrics and feature importance")
    
    def load_models(self, model_dir='../models'):
        """Load trained models from disk"""
        model_files = {
            'random_forest': 'random_forest_model.joblib',
            'xgboost': 'xgboost_model.joblib'
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(model_dir, filename)
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name} model")
        
        # Load metrics
        metrics_path = os.path.join(model_dir, 'model_metrics.joblib')
        if os.path.exists(metrics_path):
            self.model_metrics = joblib.load(metrics_path)
        
        # Load feature importance
        importance_path = os.path.join(model_dir, 'feature_importance.joblib')
        if os.path.exists(importance_path):
            self.feature_importance = joblib.load(importance_path)
    
    def predict_sales(self, model_name, X):
        """Make predictions using a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        return self.models[model_name].predict(X)

def main():
    """Main function to demonstrate the models"""
    from data_preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    data_path = '../data/ecommerce_sales_data.csv'
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(data_path)
    
    # Initialize models
    models = SalesPredictionModels()
    
    # Train models
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    # Train Random Forest
    models.train_random_forest(X_train, y_train, n_estimators=100)
    
    # Train XGBoost
    models.train_xgboost(X_train, y_train, n_estimators=100)
    
    # Evaluate models
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    models.evaluate_all_models(X_test, y_test)
    
    # Compare models
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    comparison = models.compare_models()
    
    # Create visualizations
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    # Disable interactive plotting for this run
    import matplotlib
    matplotlib.use('Agg')
    
    for model_name in models.models.keys():
        models.plot_feature_importance(model_name)
    
    models.plot_predictions_comparison(X_test, y_test)
    models.plot_residuals(X_test, y_test)
    
    # Save models
    models.save_models()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main()
