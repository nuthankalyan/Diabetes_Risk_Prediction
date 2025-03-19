import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.preprocessing import label_binarize, LabelEncoder, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
import joblib
import os
from high_accuracy_model import create_synthetic_data, train_stacking_model

def plot_feature_distributions(df):
    """Plot distributions of numerical features."""
    plt.figure(figsize=(15, 10))
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for i, col in enumerate(numerical_cols, 1):
        if i > 12:  # Limit to 12 subplots
            break
        plt.subplot(3, 4, i)
        sns.histplot(data=df, x=col, hue='CLASS', bins=30)
        plt.title(f'{col} Distribution by Class')
    plt.tight_layout()
    plt.savefig('static/predictor/images/feature_distributions.png')
    plt.close()

def plot_correlation_matrix(df):
    """Plot correlation matrix of features."""
    plt.figure(figsize=(12, 10))
    # Handle categorical data by encoding it first
    df_numeric = df.copy()
    for col in df_numeric.select_dtypes(include=['object']).columns:
        if col != 'CLASS':  # Don't encode the target yet
            df_numeric[col] = LabelEncoder().fit_transform(df_numeric[col])
    
    numerical_cols = df_numeric.select_dtypes(include=[np.number]).columns
    correlation_matrix = df_numeric[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('static/predictor/images/correlation_matrix.png')
    plt.close()

def plot_learning_curve(model, X, y):
    """Plot learning curve showing model performance with increasing data size."""
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy', 
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, label='Cross-validation score', color='green', marker='s')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
    
    plt.title('Learning Curve')
    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy Score')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('static/predictor/images/learning_curve.png')
    plt.close()

def plot_model_comparison(base_models, X_test, y_test):
    """Plot comparison of base model accuracies."""
    model_names = []
    accuracies = []
    
    for name, model in base_models:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_names.append(name)
        accuracies.append(accuracy)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies)
    plt.title('Base Model Accuracies Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('static/predictor/images/model_comparison.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('static/predictor/images/confusion_matrix.png')
    plt.close()

def plot_roc_curves(y_true, y_pred_proba, classes):
    """Plot ROC curves for each class."""
    plt.figure(figsize=(10, 8))
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('static/predictor/images/roc_curves.png')
    plt.close()

def plot_feature_importance(model, feature_names, X_test, y_test):
    """Plot feature importance."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        try:
            # For stacking model, try to extract from final estimator
            if hasattr(model, 'final_estimator_') and hasattr(model.final_estimator_, 'coef_'):
                importances = np.abs(model.final_estimator_.coef_).mean(axis=0)
            else:
                # For models without direct feature_importances_, use permutation importance
                from sklearn.inspection import permutation_importance
                importances = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42).importances_mean
        except:
            print("Could not compute feature importances")
            return
    
    # Ensure importances array is same length as feature_names
    if len(importances) != len(feature_names):
        print(f"Warning: Number of importance values ({len(importances)}) doesn't match number of features ({len(feature_names)})")
        # Trim the longer one to match the shorter one
        min_length = min(len(importances), len(feature_names))
        importances = importances[:min_length]
        feature_names = feature_names[:min_length]
    
    plt.figure(figsize=(12, 6))
    feature_names_arr = np.array(feature_names)
    indices = np.argsort(importances)[::-1]
    
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), feature_names_arr[indices], rotation=45)
    plt.tight_layout()
    plt.savefig('static/predictor/images/feature_importance.png')
    plt.close()

def plot_metrics_comparison():
    """Plot a comparison of model metrics."""
    try:
        metrics = joblib.load('model_metrics.joblib')
        
        # Create a bar chart of metrics
        plt.figure(figsize=(10, 6))
        metric_names = []
        metric_values = []
        
        # Only use numeric metrics
        for key, value in metrics.items():
            if key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'] and isinstance(value, (int, float)):
                metric_names.append(key)
                metric_values.append(value)
        
        bars = plt.bar(metric_names, metric_values)
        plt.title('Model Performance Metrics')
        plt.ylim(0, 1.1)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('static/predictor/images/metrics_comparison.png')
        plt.close()
        
        # Also save as accuracy_metrics.png for the Django template
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metric_names, metric_values)
        plt.title('Model Performance Metrics')
        plt.ylim(0, 1.1)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('static/predictor/images/accuracy_metrics.png')
        plt.close()
        
        return True
    except Exception as e:
        print(f"Could not load metrics file: {e}")
        return False

# def save_confusion_matrix_json(y_true, y_pred, classes):
#     """Save confusion matrix as JSON for the Django template."""
#     cm = confusion_matrix(y_true, y_pred)
    
#     # Create a dictionary structure matching the Django template
#     cm_dict = {
#         "matrix": [
#             [int(cm[0, 0]), int(cm[0, 1]), int(cm[0, 2])],
#             [int(cm[1, 0]), int(cm[1, 1]), int(cm[1, 2])],
#             [int(cm[2, 0]), int(cm[2, 1]), int(cm[2, 2])]
#         ]
#     }
    
#     # Save as JSON file
#     import json
#     with open('static/predictor/data/confusion_matrix.json', 'w') as f:
#         json.dump(cm_dict, f)
    
#     print("Confusion matrix saved as JSON")

def generate_all_visualizations():
    """Generate all model visualizations."""
    # Create static directory if it doesn't exist
    os.makedirs('static/predictor/images', exist_ok=True)
    os.makedirs('static/predictor/data', exist_ok=True)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    df = create_synthetic_data(3000)
    
    # Plot feature distributions
    print("Plotting feature distributions...")
    plot_feature_distributions(df)
    
    # Plot correlation matrix
    print("Plotting correlation matrix...")
    plot_correlation_matrix(df)
    
    # Create feature matrix X and target vector y
    X = df.drop('CLASS', axis=1)
    y = df['CLASS']
    
    # Encode gender and target
    print("Preprocessing data...")
    gender_encoder = LabelEncoder()
    X_processed = X.copy()
    X_processed['Gender'] = gender_encoder.fit_transform(X_processed['Gender'])
    
    class_encoder = LabelEncoder()
    y_encoded = class_encoder.fit_transform(y)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)
    
    # Scale and transform features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    power_transformer = PowerTransformer()
    X_train_transformed = power_transformer.fit_transform(X_train_scaled)
    X_test_transformed = power_transformer.transform(X_test_scaled)
    
    # Convert feature names to list
    feature_names = X_processed.columns.tolist()
    
    # Check if models directory exists, create if not
    os.makedirs('models', exist_ok=True)
    
    # Save feature columns for future reference
    joblib.dump(feature_names, 'feature_columns.joblib')
    
    # Check if models already exist
    models_exist = os.path.exists('models/stacking_model.joblib')
    
    if not models_exist:
        print("Training model...")
        # Train model
        train_stacking_model()
    
    # Load the trained model
    print("Loading models...")
    try:
        stacking_model = joblib.load('models/stacking_model.joblib')
        
        # Get predictions
        print("Making predictions...")
        y_pred = stacking_model.predict(X_test_transformed)
        y_pred_proba = stacking_model.predict_proba(X_test_transformed)
        
        # Plot learning curve (using a simpler model to avoid long computation)
        print("Plotting learning curve...")
        from sklearn.ensemble import RandomForestClassifier
        simple_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        plot_learning_curve(simple_model, X_train_transformed, y_train)
        
        # Try to load base models
        base_models = []
        for model_name in ['random_forest', 'gradient_boosting', 'xgboost', 'svm', 'neural_network']:
            try:
                model = joblib.load(f'models/{model_name}_model.joblib')
                base_models.append((model_name, model))
            except Exception as e:
                print(f"Could not load {model_name} model: {e}")
        
        # Plot model comparison if base models are available
        if base_models:
            print("Plotting model comparison...")
            plot_model_comparison(base_models, X_test_transformed, y_test)
        
        # Plot confusion matrix
        print("Plotting confusion matrix...")
        classes = class_encoder.classes_
        plot_confusion_matrix(y_test, y_pred, classes)
        
        # Save confusion matrix as JSON for Django
        # print("Saving confusion matrix as JSON...")
        # save_confusion_matrix_json(y_test, y_pred, classes)
        
        # Plot ROC curves
        print("Plotting ROC curves...")
        plot_roc_curves(y_test, y_pred_proba, classes)
        
        # Plot feature importance
        print("Plotting feature importance...")
        plot_feature_importance(stacking_model, feature_names, X_test_transformed, y_test)
        
        # Plot metrics comparison
        print("Plotting metrics comparison...")
        plot_metrics_comparison()
        
        print("All visualizations have been generated successfully!")
        print("The plots are saved in the static/predictor/images/ directory")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        print(traceback.format_exc())
        print("Some visualizations could not be generated.")

if __name__ == "__main__":
    generate_all_visualizations() 