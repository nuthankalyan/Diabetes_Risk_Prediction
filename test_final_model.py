import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import logging
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_components():
    """Load all model components needed for prediction"""
    logger.info("Loading model components...")
    
    # Load model files
    try:
        model = joblib.load('ensemble_data.joblib')
        class_encoder = joblib.load('class_encoder.joblib')
        gender_encoder = joblib.load('gender_encoder.joblib')
        scaler = joblib.load('scaler.joblib')
        power_transformer = joblib.load('power_transformer.joblib')
        feature_selector = joblib.load('feature_selector.joblib')
        feature_columns = joblib.load('feature_columns.joblib')
        
        logger.info("Model components loaded successfully")
        return {
            'model': model,
            'class_encoder': class_encoder,
            'gender_encoder': gender_encoder,
            'scaler': scaler,
            'power_transformer': power_transformer,
            'feature_selector': feature_selector,
            'feature_columns': feature_columns
        }
    except Exception as e:
        logger.error(f"Error loading model components: {e}")
        raise

def create_derived_features(df):
    """Create all derived features needed for the model"""
    logger.info("Creating derived features...")
    
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Convert column names to match the model training names
    column_mapping = {
        'Gender': 'Gender',
        'Age': 'Age',
        'Urea': 'Urea',
        'Cr': 'Cr',
        'HbA1c': 'HbA1c',
        'Chol': 'Chol',
        'TG': 'TG',
        'HDL': 'HDL',
        'LDL': 'LDL',
        'VLDL': 'VLDL',
        'BMI': 'BMI'
    }
    
    df_processed.rename(columns=column_mapping, inplace=True)
    
    # Encode gender
    df_processed['Gender_Encoded'] = components['gender_encoder'].transform(df_processed['Gender'])
    
    # Create derived features
    df_processed['BMI_Age_Ratio'] = df_processed['BMI'] / df_processed['Age']
    df_processed['HbA1c_BMI_Ratio'] = df_processed['HbA1c'] / df_processed['BMI']
    df_processed['HbA1c_BMI'] = df_processed['HbA1c'] * df_processed['BMI']
    df_processed['Chol_HDL_Ratio'] = df_processed['Chol'] / df_processed['HDL']
    df_processed['TG_HDL_Ratio'] = df_processed['TG'] / df_processed['HDL']
    df_processed['LDL_HDL_Ratio'] = df_processed['LDL'] / df_processed['HDL']
    
    # Log transformations
    df_processed['Log_HbA1c'] = np.log1p(df_processed['HbA1c'])
    df_processed['Log_Chol'] = np.log1p(df_processed['Chol'])
    df_processed['Log_TG'] = np.log1p(df_processed['TG'])
    df_processed['Log_BMI'] = np.log1p(df_processed['BMI'])
    df_processed['Log_Urea'] = np.log1p(df_processed['Urea'])
    df_processed['Log_Cr'] = np.log1p(df_processed['Cr'])
    
    # Squared terms
    df_processed['HbA1c_Squared'] = df_processed['HbA1c'] ** 2
    df_processed['BMI_Squared'] = df_processed['BMI'] ** 2
    
    # Composite scores
    df_processed['Metabolic_Score'] = (
        df_processed['HbA1c'] * 0.3 + 
        df_processed['BMI'] * 0.2 + 
        df_processed['Chol'] * 0.1 + 
        df_processed['TG'] * 0.1 + 
        df_processed['LDL'] * 0.1 - 
        df_processed['HDL'] * 0.2
    )
    
    df_processed['Kidney_Score'] = (
        df_processed['Urea'] * 0.6 + 
        df_processed['Cr'] * 0.4
    )
    
    df_processed['Lipid_Score'] = (
        df_processed['Chol'] * 0.25 + 
        df_processed['TG'] * 0.25 + 
        df_processed['LDL'] * 0.25 - 
        df_processed['HDL'] * 0.25
    )
    
    # Age group (categorical)
    df_processed['Age_Group'] = pd.cut(
        df_processed['Age'], 
        bins=[0, 30, 45, 60, 100], 
        labels=[0, 1, 2, 3]
    ).astype(int)
    
    # BMI category (categorical)
    df_processed['BMI_Category'] = pd.cut(
        df_processed['BMI'], 
        bins=[0, 18.5, 25, 30, 100], 
        labels=[0, 1, 2, 3]
    ).astype(int)
    
    # HbA1c category (categorical)
    df_processed['HbA1c_Category'] = pd.cut(
        df_processed['HbA1c'], 
        bins=[0, 5.7, 6.5, 100], 
        labels=[0, 1, 2]
    ).astype(int)
    
    logger.info(f"Created {len(df_processed.columns) - len(df.columns)} derived features")
    
    # Return the dataframe with all required features
    return df_processed

def test_model_accuracy():
    """Test the final model's accuracy on the test dataset"""
    logger.info("Testing final model accuracy...")
    
    # Load model components
    global components  # Make components available to create_derived_features
    components = load_model_components()
    feature_columns = components['feature_columns']
    logger.info(f"Original feature columns: {feature_columns}")
    
    # Load the individual models
    models_dir = "models"
    model_files = {
        "Random Forest": os.path.join(models_dir, "random_forest_model.joblib"),
        "Gradient Boosting": os.path.join(models_dir, "gradient_boosting_model.joblib"),
        "XGBoost": os.path.join(models_dir, "xgboost_model.joblib"),
        "SVM": os.path.join(models_dir, "svm_model.joblib"),
        "Neural Network": os.path.join(models_dir, "neural_network_model.joblib")
    }
    
    # Load models
    models = {}
    ensemble_config = components['model']
    model_names = ensemble_config['names']
    model_weights = ensemble_config['weights']
    
    for name, path in model_files.items():
        if name in model_names and os.path.exists(path):
            logger.info(f"Loading {name} model from {path}")
            models[name] = joblib.load(path)
        else:
            logger.warning(f"Model {name} not found at {path}")
    
    # Verify we have all required models
    if not all(name in models for name in model_names):
        missing = [name for name in model_names if name not in models]
        logger.error(f"Missing models: {missing}")
        raise FileNotFoundError(f"Missing required models: {missing}")
    
    # Load the test dataset
    try:
        df = pd.read_csv('Balanced_diabetes_dataset.csv')
        logger.info(f"Loaded test dataset with {len(df)} samples")
    except Exception as e:
        logger.error(f"Error loading test dataset: {e}")
        raise
    
    # Extract target
    y = df['CLASS']
    
    # Preprocess the data and create derived features
    X_all = df.drop(['CLASS', 'ID'], axis=1)
    X_processed = create_derived_features(X_all)
    
    # Select only the columns used by the model
    logger.info(f"Required features: {feature_columns}")
    logger.info(f"Available features: {X_processed.columns.tolist()}")
    
    # Ensure all required features are present
    missing_cols = set(feature_columns) - set(X_processed.columns)
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        raise ValueError(f"Missing required features: {missing_cols}")
    
    # Select only the features needed by the model
    X = X_processed[feature_columns]
    
    # Verify feature count
    logger.info(f"Feature count: {X.shape[1]} (expected 30)")
    
    # Apply power transformation
    X_transformed = components['power_transformer'].transform(X)
    X_transformed_df = pd.DataFrame(X_transformed, columns=X.columns)
    
    # Scale features
    X_scaled = components['scaler'].transform(X_transformed_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Select features
    X_selected = components['feature_selector'].transform(X_scaled_df)
    
    # Make predictions with each model
    predictions = {}
    probas = {}
    
    for name in model_names:
        model = models[name]
        predictions[name] = model.predict(X_selected)
        probas[name] = model.predict_proba(X_selected)
    
    # Combine predictions using weighted voting
    y_pred_proba = np.zeros((len(X_selected), 3))  # Assuming 3 classes
    
    for i, name in enumerate(model_names):
        weight = model_weights[i]
        y_pred_proba += weight * probas[name]
    
    # Get final class prediction
    y_pred = [components['class_encoder'].classes_[i] for i in np.argmax(y_pred_proba, axis=1)]
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    
    # Calculate ROC AUC
    y_one_hot = pd.get_dummies(y)
    roc_auc = roc_auc_score(y_one_hot, y_pred_proba, multi_class='ovr')
    
    # Print results
    logger.info(f"Final Model Accuracy: {accuracy:.4f}")
    logger.info(f"Final Model Precision: {precision:.4f}")
    logger.info(f"Final Model Recall: {recall:.4f}")
    logger.info(f"Final Model F1 Score: {f1:.4f}")
    logger.info(f"Final Model ROC AUC: {roc_auc:.4f}")
    
    # Print classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=components['class_encoder'].classes_,
                yticklabels=components['class_encoder'].classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs('static/images', exist_ok=True)
    
    plt.savefig('static/images/final_confusion_matrix.png')
    logger.info("Confusion matrix saved to static/images/final_confusion_matrix.png")
    
    # Plot ROC Curve for each class
    plt.figure(figsize=(10, 8))
    
    if y_pred_proba.shape[1] == 3:  # Multi-class
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        
        classes = components['class_encoder'].classes_
        y_bin = label_binarize(y, classes=classes)
        
        # ROC curve for each class
        for i, class_name in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, 
                    label=f'ROC curve (class: {class_name}) (area = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('static/images/final_roc_curve.png')
    logger.info("ROC curve saved to static/images/final_roc_curve.png")
    
    # Save metrics to file
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    joblib.dump(metrics, 'model_metrics.joblib')
    logger.info("Model metrics saved to model_metrics.joblib")
    
    return metrics

if __name__ == "__main__":
    # Create static/images directory if it doesn't exist
    os.makedirs('static/images', exist_ok=True)
    
    # Test the model
    metrics = test_model_accuracy()
    
    # Print summary
    print("\n" + "="*50)
    print("FINAL MODEL PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print("="*50) 