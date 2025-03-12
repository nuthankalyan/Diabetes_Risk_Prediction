#!/usr/bin/env python
"""
Script to pre-train the diabetes prediction model and save it to the repository.
This can be useful if model training is failing during deployment.
"""

import os
import sys
import logging

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
        
        # Train the model
        model, scaler = train_model()
        
        logger.info("Model pre-training completed successfully!")
        logger.info("Model files saved to predictor/models/")
        logger.info("You can now commit these files to your repository.")
        
        # List the model files
        model_files = os.listdir('predictor/models')
        logger.info(f"Model files: {model_files}")
        
        return 0
    except Exception as e:
        logger.error(f"Error during model pre-training: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 