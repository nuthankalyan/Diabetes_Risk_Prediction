from django.shortcuts import render
from .forms import DiabetesPredictionForm
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

def home(request):
    prediction_result = None
    probability = None
    form = DiabetesPredictionForm()

    if request.method == 'POST':
        form = DiabetesPredictionForm(request.POST)
        if form.is_valid():
            # Get features from form
            features = [
                float(form.cleaned_data['Age']),
                float(form.cleaned_data['Pregnancies']),
                float(form.cleaned_data['BMI']),
                float(form.cleaned_data['Glucose']),
                float(form.cleaned_data['BloodPressure']),
                float(form.cleaned_data['HbA1c']),
                float(form.cleaned_data['LDL']),
                float(form.cleaned_data['HDL']),
                float(form.cleaned_data['Triglycerides']),
                float(form.cleaned_data['WaistCircumference']),
                float(form.cleaned_data['HipCircumference']),
                float(form.cleaned_data['WHR']),
                float(form.cleaned_data['FamilyHistory']),
                float(form.cleaned_data['DietType']),
                float(form.cleaned_data['Hypertension']),
                float(form.cleaned_data['MedicationUse'])
            ]
            
            try:
                # Get prediction
                prediction, prob = predict_diabetes(features)
                prediction_result = "Positive (High Risk)" if prediction == 1 else "Negative (Low Risk)"
                probability = round(prob[1] * 100, 2) if prediction == 1 else round(prob[0] * 100, 2)
            except Exception as e:
                prediction_result = "Error in prediction"
                probability = None

    return render(request, 'predictor/home.html', {
        'form': form,
        'prediction': prediction_result,
        'probability': probability
    })

def visualizations(request):
    # Get model metrics and visualizations
    metrics, feature_importance, correlation_matrix, confusion_mat, roc_data = get_model_metrics()
    
    # Convert plots to base64 for embedding in HTML
    feature_importance_plot = get_feature_importance_plot(feature_importance)
    correlation_plot = get_correlation_plot(correlation_matrix)
    confusion_matrix_plot = get_confusion_matrix_plot(confusion_mat)
    roc_curve_plot = get_roc_curve_plot(roc_data)
    
    return render(request, 'predictor/visualizations.html', {
        'metrics': metrics,
        'feature_importance_plot': feature_importance_plot,
        'correlation_plot': correlation_plot,
        'confusion_matrix_plot': confusion_matrix_plot,
        'roc_curve_plot': roc_curve_plot
    })

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
