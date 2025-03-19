import joblib
import json
import os

# Load the metrics file
metrics = joblib.load('model_metrics.joblib')

# Load the confusion matrix from JSON
with open('static/predictor/data/confusion_matrix.json', 'r') as f:
    cm = json.load(f)

# Update the metrics with the confusion matrix
metrics['confusion_matrix'] = cm

# Save the updated metrics
joblib.dump(metrics, 'model_metrics.joblib')

# Also copy the updated metrics file to the predictor directory
if not os.path.exists('predictor'):
    os.makedirs('predictor', exist_ok=True)

joblib.dump(metrics, 'predictor/model_metrics.joblib')

print('Updated model_metrics.joblib with confusion matrix')
print('Copied updated metrics to predictor directory') 