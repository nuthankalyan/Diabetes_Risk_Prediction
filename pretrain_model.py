#!/usr/bin/env python
"""
Script to pre-train the diabetes prediction model and save it to the repository.
This ensures consistent model visualizations across different environments.
"""

import os
import sys
import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # Get the absolute path to the project root
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        project_root = script_dir
        
        # Add the project root to the path so we can import from the predictor package
        sys.path.append(str(project_root))
        
        # Import the train_model function from ml_model.py
        from predictor.ml_model import train_model
        
        logger.info(f"Project root: {project_root}")
        logger.info("Starting pre-training of the diabetes prediction model...")
        
        # Get the absolute path to the models directory
        models_dir = os.path.join(project_root, 'predictor', 'models')
        logger.info(f"Models directory: {models_dir}")
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Define model file paths
        model_files = [
            os.path.join(models_dir, 'stacking_model.joblib'),
            os.path.join(models_dir, 'scaler.joblib'),
            os.path.join(models_dir, 'model_data.joblib')
        ]
        
        # Backup existing model files if they exist
        backup_dir = os.path.join(models_dir, 'backup')
        os.makedirs(backup_dir, exist_ok=True)
        
        for file in model_files:
            if os.path.exists(file):
                backup_file = os.path.join(backup_dir, os.path.basename(file))
                logger.info(f"Backing up {file} to {backup_file}")
                shutil.copy2(file, backup_file)
        
        # Train the model
        model, scaler = train_model()
        
        logger.info("Model pre-training completed successfully!")
        logger.info(f"Model files saved to {models_dir}")
        
        # List the model files
        model_files = [f for f in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, f))]
        logger.info(f"Model files: {model_files}")
        
        logger.info("\nIMPORTANT: To ensure consistent visualizations across environments:")
        logger.info("1. Commit these model files to your repository:")
        logger.info(f"   git add {os.path.join('predictor', 'models', '*.joblib')}")
        logger.info("   git commit -m \"Add pre-trained model files for consistent visualizations\"")
        logger.info("   git push")
        logger.info("2. Deploy the updated repository to Render")
        logger.info("3. This will ensure both local and deployed environments use the same model")
        
        return 0
    except Exception as e:
        logger.error(f"Error during model pre-training: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 