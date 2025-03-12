#!/usr/bin/env python
"""
Debug script to verify model loading and visualizations.
Run this script to check if the model is being loaded correctly and to generate
visualization data for comparison between environments.
"""

import os
import sys
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_file_hash(filepath):
    """Calculate the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    try:
        # Get the absolute path to the project root
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        project_root = script_dir
        
        # Add the project root to the path so we can import from the predictor package
        sys.path.append(str(project_root))
        
        logger.info(f"Project root: {project_root}")
        
        # Get the absolute path to the models directory
        models_dir = os.path.join(project_root, 'predictor', 'models')
        logger.info(f"Models directory: {models_dir}")
        
        # Define model file paths
        model_path = os.path.join(models_dir, 'stacking_model.joblib')
        scaler_path = os.path.join(models_dir, 'scaler.joblib')
        model_data_path = os.path.join(models_dir, 'model_data.joblib')
        
        # Check if model files exist
        model_exists = os.path.exists(model_path)
        scaler_exists = os.path.exists(scaler_path)
        model_data_exists = os.path.exists(model_data_path)
        
        logger.info(f"Model file exists: {model_exists} ({model_path})")
        logger.info(f"Scaler file exists: {scaler_exists} ({scaler_path})")
        logger.info(f"Model data file exists: {model_data_exists} ({model_data_path})")
        
        if not model_exists or not scaler_exists or not model_data_exists:
            logger.error("One or more model files are missing. Please run pretrain_model.py first.")
            return 1
        
        # Calculate file hashes
        model_hash = calculate_file_hash(model_path)
        scaler_hash = calculate_file_hash(scaler_path)
        model_data_hash = calculate_file_hash(model_data_path)
        
        logger.info(f"Model file hash: {model_hash}")
        logger.info(f"Scaler file hash: {scaler_hash}")
        logger.info(f"Model data file hash: {model_data_hash}")
        
        # Load model data
        logger.info("Loading model data...")
        model = joblib.load(model_path)
        model_data = joblib.load(model_data_path)
        
        # Extract data for visualizations
        X_train = model_data['X_train']
        X_test = model_data['X_test']
        y_test = model_data['y_test']
        y_pred = model_data['y_pred']
        feature_names = model_data['feature_names']
        
        # Get feature importance
        rf_model = model.estimators_[0][1]  # Get the Random Forest model
        importances = rf_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # Print top 5 features
        logger.info("Top 5 features by importance:")
        for i, (feature, importance) in enumerate(zip(feature_importance['feature'].head(5), feature_importance['importance'].head(5))):
            logger.info(f"{i+1}. {feature}: {importance:.4f}")
        
        # Calculate correlation matrix
        correlation_matrix = X_train.corr()
        
        # Print top 5 correlations
        logger.info("Top 5 feature correlations:")
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for i, (feat1, feat2, corr) in enumerate(corr_pairs[:5]):
            logger.info(f"{i+1}. {feat1} - {feat2}: {corr:.4f}")
        
        # Calculate confusion matrix
        conf_matrix = np.array([[sum((y_test == 0) & (y_pred == 0)), sum((y_test == 0) & (y_pred == 1))],
                               [sum((y_test == 1) & (y_pred == 0)), sum((y_test == 1) & (y_pred == 1))]])
        
        logger.info("Confusion Matrix:")
        logger.info(f"TN: {conf_matrix[0, 0]}, FP: {conf_matrix[0, 1]}")
        logger.info(f"FN: {conf_matrix[1, 0]}, TP: {conf_matrix[1, 1]}")
        
        # Calculate metrics
        accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / conf_matrix.sum()
        precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1]) if (conf_matrix[1, 1] + conf_matrix[0, 1]) > 0 else 0
        recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]) if (conf_matrix[1, 1] + conf_matrix[1, 0]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        logger.info("Model Metrics:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        # Save debug info to file
        debug_info = {
            'model_hash': model_hash,
            'scaler_hash': scaler_hash,
            'model_data_hash': model_data_hash,
            'top_features': feature_importance.head(5).to_dict('records'),
            'top_correlations': [{'feature1': feat1, 'feature2': feat2, 'correlation': float(corr)} for feat1, feat2, corr in corr_pairs[:5]],
            'confusion_matrix': conf_matrix.tolist(),
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
        }
        
        debug_file = os.path.join(project_root, 'model_debug_info.json')
        with open(debug_file, 'w') as f:
            json.dump(debug_info, f, indent=2)
        
        logger.info(f"Debug information saved to {debug_file}")
        logger.info("Run this script in both environments (local and Render) and compare the output files to identify differences.")
        
        return 0
    except Exception as e:
        logger.error(f"Error during model debugging: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 