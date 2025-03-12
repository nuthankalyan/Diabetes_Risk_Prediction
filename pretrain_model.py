#!/usr/bin/env python
"""
Script to pre-train the diabetes prediction model and save it to the repository.
This ensures consistent model visualizations across different environments.
"""

import os
import sys
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # Add the current directory to the path so we can import from the predictor package
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Import the train_model function from ml_model.py
        from predictor.ml_model import train_model
        
        logger.info("Starting pre-training of the diabetes prediction model...")
        
        # Create models directory if it doesn't exist
        os.makedirs('predictor/models', exist_ok=True)
        
        # Backup existing model files if they exist
        model_files = [
            'predictor/models/stacking_model.joblib',
            'predictor/models/scaler.joblib',
            'predictor/models/model_data.joblib'
        ]
        
        backup_dir = 'predictor/models/backup'
        os.makedirs(backup_dir, exist_ok=True)
        
        for file in model_files:
            if os.path.exists(file):
                backup_file = os.path.join(backup_dir, os.path.basename(file))
                logger.info(f"Backing up {file} to {backup_file}")
                shutil.copy2(file, backup_file)
        
        # Train the model
        model, scaler = train_model()
        
        logger.info("Model pre-training completed successfully!")
        logger.info("Model files saved to predictor/models/")
        
        # List the model files
        model_files = os.listdir('predictor/models')
        model_files = [f for f in model_files if f != 'backup']
        logger.info(f"Model files: {model_files}")
        
        logger.info("\nIMPORTANT: To ensure consistent visualizations across environments:")
        logger.info("1. Commit these model files to your repository:")
        logger.info("   git add predictor/models/*.joblib")
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