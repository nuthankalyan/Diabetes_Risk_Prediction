from django.shortcuts import render
from django.contrib import messages
import logging
from .ml_model import predict_diabetes, get_model_metrics
import numpy as np
import pandas as pd
# Set matplotlib to use a non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend which doesn't require a GUI
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def home(request):
    """Render the home page with the prediction form."""
    return render(request, 'predictor/home.html')

def predict(request):
    """Process form submission and return prediction results."""
    if request.method == 'POST':
        try:
            # Extract features from form
            features = {
                'gender': request.POST.get('gender'),
                'age': float(request.POST.get('age')),
                'bmi': float(request.POST.get('bmi')),
                'urea': 30.0,  # Default values for now
                'cr': 0.9,
                'hba1c': float(request.POST.get('hba1c')),
                'chol': float(request.POST.get('cholesterol')),  # Map 'cholesterol' from form to 'chol' expected by model
                'tg': 150.0,  # Default value
                'hdl': float(request.POST.get('hdl')),
                'ldl': 120.0,  # Default value
                'vldl': 25.0,  # Default value
            }
            
            logger.info(f"Processing input features: {features}")
            
            # Make prediction
            prediction, probability, risk_factors, risk_details = predict_diabetes(features)
            
            logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}")
            
            # Prepare context for template
            context = {
                'prediction': prediction,
                'probability': f"{probability:.1%}",
                'risk_factors': risk_factors,
                'risk_details': risk_details,
                'features': features
            }
            
            return render(request, 'predictor/result.html', context)
            
        except ValueError as e:
            messages.error(request, f"Invalid input: {str(e)}")
            logger.error(f"Error processing form data: {str(e)}")
        except Exception as e:
            messages.error(request, "An error occurred while processing your request.")
            logger.error(f"Error making prediction: {str(e)}")
    
    return render(request, 'predictor/predict.html')

def visualizations(request):
    """Render the visualizations page."""
    # Get model metrics for the visualizations page
    metrics = get_model_metrics()
    return render(request, 'predictor/visualizations.html', {'metrics': metrics})

def get_feature_importance_plot(feature_importance):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.sort_values('importance', ascending=False))
    plt.title('Feature Importance')
    plt.tight_layout()
    
    return get_plot_as_base64(plt)

def get_correlation_plot(correlation_matrix):
    plt.figure(figsize=(12, 10))
    mask = np.triu(correlation_matrix)
    sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    return get_plot_as_base64(plt)

def get_confusion_matrix_plot(confusion_mat):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    return get_plot_as_base64(plt)

def get_roc_curve_plot(roc_data):
    fpr, tpr, roc_auc = roc_data
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    return get_plot_as_base64(plt)

def get_plot_as_base64(plt):
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return base64.b64encode(image_png).decode('utf-8')
