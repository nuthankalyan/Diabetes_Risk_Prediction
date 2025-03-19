from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import logging
from predictor.ml_model import predict_diabetes, get_model_metrics
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
            # Extract features from the form
            features = {
                'gender': request.POST.get('gender', ''),
                'age': request.POST.get('age', 0),
                'urea': request.POST.get('urea', 0),
                'cr': request.POST.get('cr', 0),
                'hba1c': request.POST.get('hba1c', 0),
                'chol': request.POST.get('chol', 0),
                'tg': request.POST.get('tg', 0),
                'hdl': request.POST.get('hdl', 0),
                'ldl': request.POST.get('ldl', 0),
                'vldl': request.POST.get('vldl', 0),
                'bmi': request.POST.get('bmi', 0)
            }
            
            logger.info(f"Processing input features: {features}")
            
            # Make prediction
            prediction, probability, risk_factors, risk_details = predict_diabetes(features)
            
            # Prepare context for the template
            context = {
                'prediction': prediction,
                'probability': f"{probability:.2%}",
                'probability_value': probability,
                'risk_factors': risk_factors,
                'risk_details': risk_details
            }
            
            logger.info(f"Prediction: {prediction}, Probability: {probability:.2%}")
            return render(request, 'predictor/home.html', context)
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            context = {'error': f"An error occurred: {str(e)}"}
            return render(request, 'predictor/home.html', context)
    
    # If not POST, redirect to home
    return render(request, 'predictor/home.html')

def visualizations(request):
    """Render the visualizations page."""
    # Get model metrics for the visualizations page
    metrics = get_model_metrics()
    context = {'metrics': metrics}
    return render(request, 'predictor/visualizations.html', context)

@csrf_exempt
def api_predict(request):
    """
    REST API endpoint for making diabetes predictions.
    Accepts JSON post data and returns prediction results.
    """
    if request.method == 'POST':
        try:
            # Parse JSON data from request body
            data = json.loads(request.body)
            
            # Extract features from the JSON
            features = {
                'gender': data.get('gender', 'M'),
                'age': float(data.get('age', 0)),
                'urea': float(data.get('urea', 0)),
                'cr': float(data.get('cr', 0)),
                'hba1c': float(data.get('hba1c', 0)),
                'chol': float(data.get('chol', 0)),
                'tg': float(data.get('tg', 0)),
                'hdl': float(data.get('hdl', 0)),
                'ldl': float(data.get('ldl', 0)),
                'vldl': float(data.get('vldl', 0)),
                'bmi': float(data.get('bmi', 0))
            }
            
            logger.info(f"API: Processing input features: {features}")
            
            # Make prediction
            prediction, probability, risk_factors, risk_details = predict_diabetes(features)
            
            # Prepare JSON response
            response_data = {
                'success': True,
                'prediction': prediction,
                'probability': f"{probability:.4f}",
                'probability_raw': float(probability),
                'risk_factors': risk_factors,
                'risk_details': risk_details
            }
            
            logger.info(f"API: Prediction: {prediction}, Probability: {probability:.4f}")
            return JsonResponse(response_data)
            
        except json.JSONDecodeError:
            logger.error("API: Invalid JSON data in request")
            return JsonResponse({
                'success': False,
                'error': 'Invalid JSON data in request body'
            }, status=400)
            
        except Exception as e:
            logger.error(f"API: Error making prediction: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': f"An error occurred: {str(e)}"
            }, status=500)
    
    # If not POST method
    return JsonResponse({
        'success': False,
        'error': 'Only POST method is supported for this endpoint'
    }, status=405)

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
