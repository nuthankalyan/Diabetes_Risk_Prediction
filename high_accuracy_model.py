import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_synthetic_data(n_samples=1000):
    """Create synthetic diabetes dataset."""
    np.random.seed(42)
    
    # Generate synthetic data
    data = {
        'Gender': np.random.choice(['M', 'F'], n_samples),
        'Age': np.random.normal(45, 15, n_samples),
        'BMI': np.random.normal(28, 6, n_samples),
        'HbA1c': np.random.normal(6.0, 1.5, n_samples),
        'Cholesterol': np.random.normal(200, 40, n_samples),
        'HDL': np.random.normal(50, 15, n_samples),
        'Urea': np.random.normal(30, 10, n_samples),
        'Cr': np.random.normal(0.9, 0.3, n_samples),
        'TG': np.random.normal(150, 50, n_samples),
        'LDL': np.random.normal(120, 30, n_samples),
        'VLDL': np.random.normal(25, 8, n_samples)
    }
    
    # Create diabetes risk based on features
    risk_score = (
        (data['HbA1c'] > 6.5).astype(int) * 3 +
        (data['BMI'] > 30).astype(int) * 2 +
        (data['Age'] > 45).astype(int) +
        (data['Cholesterol'] > 240).astype(int) +
        (data['HDL'] < 40).astype(int)
    )
    
    data['CLASS'] = np.where(risk_score >= 4, 'Yes',
                            np.where(risk_score >= 2, 'Possible', 'No'))
    
    return pd.DataFrame(data)

def train_models():
    """Train and save the ensemble model."""
    try:
        # Create synthetic dataset
        logger.info("Creating synthetic dataset...")
        df = create_synthetic_data(2000)
        
        # Create directories if they don't exist
        os.makedirs('models', exist_ok=True)
        
        # Prepare features and target
        X = df.drop('CLASS', axis=1)
        y = df['CLASS']
        
        # Encode categorical variables
        gender_encoder = LabelEncoder()
        X['Gender'] = gender_encoder.fit_transform(X['Gender'])
        
        class_encoder = LabelEncoder()
        y = class_encoder.fit_transform(y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Apply power transformation
        power_transformer = PowerTransformer()
        X_train_transformed = power_transformer.fit_transform(X_train_scaled)
        X_test_transformed = power_transformer.transform(X_test_scaled)
        
        # Initialize models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        }
        
        # Train models and get predictions
        weights = {
            'SVM': 0.30,
            'Neural Network': 0.25,
            'Random Forest': 0.15,
            'Gradient Boosting': 0.15,
            'XGBoost': 0.15
        }
        
        ensemble_preds = np.zeros((X_test_transformed.shape[0], len(np.unique(y))))
        
        # Train and save each model
        for name, model in models.items():
            logger.info(f"Training {name} model...")
            model.fit(X_train_transformed, y_train)
            
            # Save model
            model_filename = f"models/{name.lower().replace(' ', '_')}_model.joblib"
            joblib.dump(model, model_filename)
            logger.info(f"Saved {name} model to {model_filename}")
            
            # Get predictions
            pred_proba = model.predict_proba(X_test_transformed)
            ensemble_preds += weights[name] * pred_proba
        
        # Get ensemble predictions
        y_pred = np.argmax(ensemble_preds, axis=1)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, ensemble_preds, multi_class='ovr')
        }
        
        # Save artifacts
        joblib.dump(gender_encoder, 'gender_encoder.joblib')
        joblib.dump(class_encoder, 'class_encoder.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        joblib.dump(power_transformer, 'power_transformer.joblib')
        joblib.dump(metrics, 'model_metrics.joblib')
        joblib.dump({'names': list(weights.keys()), 'weights': list(weights.values())},
                   'ensemble_data.joblib')
        joblib.dump(X.columns.tolist(), 'feature_columns.joblib')
        
        logger.info("Model training completed successfully!")
        logger.info(f"Model metrics: {metrics}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return False

if __name__ == "__main__":
    train_models() 