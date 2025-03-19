#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Collect static files
python manage.py collectstatic --no-input

# Apply database migrations
python manage.py migrate

# Check if model files exist
if [ -f "predictor/diabetes_model.joblib" ] && [ -f "predictor/scaler.joblib" ] && [ -f "predictor/model_metrics.joblib" ]; then
    echo "Model files found. Skipping model training to ensure consistent visualizations."
    echo "Model files:"
    ls -la predictor/*.joblib
else
    echo "Model files not found. Creating balanced dataset and calculating accuracy..."
    # Generate balanced data and train the model
    python create_balanced_dataset.py
    python calculate_accuracy.py
    python plot_accuracy.py
    # Copy files to predictor directory if needed
    cp -n *.joblib predictor/
    
    # Create static images directory if it doesn't exist
    mkdir -p predictor/static/images/
    cp -n accuracy_metrics.png predictor/static/images/
    cp -n per_class_metrics.png predictor/static/images/
    echo "Model training completed. Model files:"
    ls -la predictor/*.joblib
fi 