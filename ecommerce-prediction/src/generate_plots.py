"""
Generate plots for trained models
"""

from models import SalesPredictionModels
from data_preprocessing import DataPreprocessor
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Load and preprocess data
preprocessor = DataPreprocessor()
data_path = '../data/ecommerce_sales_data.csv'
X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(data_path)

# Initialize and train models
models = SalesPredictionModels()
models.train_random_forest(X_train, y_train, n_estimators=100)
models.train_xgboost(X_train, y_train, n_estimators=100)

# Evaluate models
models.evaluate_all_models(X_test, y_test)

# Generate plots without showing them
print("Generating plots...")

# Feature importance plots
for model_name in models.models.keys():
    models.plot_feature_importance(model_name)

# Model comparison plot
comparison_df = models.model_metrics
import pandas as pd
import matplotlib.pyplot as plt
import os

comparison_df = pd.DataFrame(comparison_df).T
comparison_df = comparison_df.round(4)

# Plot comparison
metrics_to_plot = ['R2', 'Accuracy']
fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(12, 5))

for i, metric in enumerate(metrics_to_plot):
    comparison_df[metric].plot(kind='bar', ax=axes[i])
    axes[i].set_title(f'{metric} Comparison')
    axes[i].set_ylabel(metric)
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
os.makedirs('../models/plots', exist_ok=True)
plt.savefig('../models/plots/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Predictions comparison plot (simplified version)
predictions = {}
for model_name in models.models.keys():
    _, y_pred = models.evaluate_model(model_name, X_test, y_test)
    predictions[model_name] = y_pred

# Sample data for better visualization
import numpy as np
sample_size = 1000
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
    axes[i].set_title(f'{model_name.upper()}\nR² = {models.model_metrics[model_name]["R2"]:.3f}')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../models/plots/predictions_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Save models
models.save_models()

print("All plots generated successfully!")
print("Models saved!")
