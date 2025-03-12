#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Collect static files
python manage.py collectstatic --no-input

# Apply database migrations
python manage.py migrate

# Create models directory if it doesn't exist
mkdir -p predictor/models

# Check if pre-trained model files exist
if [ -f "predictor/models/stacking_model.joblib" ] && [ -f "predictor/models/scaler.joblib" ] && [ -f "predictor/models/model_data.joblib" ]; then
    echo "Pre-trained model files found. Skipping model training to ensure consistent visualizations."
    echo "Model files:"
    ls -la predictor/models/*.joblib
else
    echo "Pre-trained model files not found. Training the model..."
    # Train the model
    python predictor/ml_model.py
    echo "Model training completed. Model files:"
    ls -la predictor/models/*.joblib
fi 