import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import subprocess
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the absolute path of the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

def train_model():
    """Train the diabetes prediction model using the high accuracy model script."""
    try:
        # Get parent directory to locate the dataset and script
        parent_dir = os.path.dirname(current_dir)
        
        # Path to high accuracy model script
        high_accuracy_script = os.path.join(parent_dir, 'high_accuracy_model.py')
        
        if not os.path.exists(high_accuracy_script):
            logger.error(f"High accuracy model script not found at {high_accuracy_script}")
            return False
            
        # Run the high accuracy model script
        logger.info(f"Running high accuracy model training script: {high_accuracy_script}")
        subprocess.run([sys.executable, high_accuracy_script], check=True)
        
        # Copy model files to predictor directory
        model_files = [
            'gender_encoder.joblib',
            'class_encoder.joblib',
            'scaler.joblib',
            'power_transformer.joblib',
            'feature_selector.joblib',
            'feature_columns.joblib',
            'model_metrics.joblib',
            'ensemble_data.joblib'
        ]
        
        # Also copy the model files from models directory
        models_dir = os.path.join(parent_dir, 'models')
        if os.path.exists(models_dir):
            model_names = [
                'random_forest_model.joblib',
                'gradient_boosting_model.joblib',
                'xgboost_model.joblib',
                'svm_model.joblib',
                'neural_network_model.joblib'
            ]
            
            # Create models directory in predictor if it doesn't exist
            predictor_models_dir = os.path.join(current_dir, 'models')
            os.makedirs(predictor_models_dir, exist_ok=True)
            
            # Copy model files to predictor models directory
            for model_name in model_names:
                src_path = os.path.join(models_dir, model_name)
                dest_path = os.path.join(predictor_models_dir, model_name)
                if os.path.exists(src_path):
                    logger.info(f"Copying {model_name} to predictor models directory")
                    try:
                        import shutil
                        shutil.copy2(src_path, dest_path)
                    except Exception as e:
                        logger.error(f"Error copying {model_name}: {str(e)}")
        
        # Copy base files to predictor directory
        for file_name in model_files:
            src_path = os.path.join(parent_dir, file_name)
            dest_path = os.path.join(current_dir, file_name)
            if os.path.exists(src_path):
                logger.info(f"Copying {file_name} to predictor directory")
                try:
                    import shutil
                    shutil.copy2(src_path, dest_path)
                except Exception as e:
                    logger.error(f"Error copying {file_name}: {str(e)}")
        
        logger.info("High accuracy model training and file copying completed")
        return True
    
    except Exception as e:
        logger.error(f"Error training high accuracy model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def predict_diabetes(features):
    """
    Predict diabetes using the trained stacking ensemble model.
    
    Args:
        features (dict): Dictionary containing patient features.
                         Expected keys: 'gender', 'age', 'urea', 'cr', 'hba1c', 
                         'chol', 'tg', 'hdl', 'ldl', 'vldl', 'bmi'
    
    Returns:
        tuple: (prediction, probability, risk_factors, risk_details)
    """
    try:
        # Log input features
        logger.info(f"predict_diabetes called with features: {features}")
        
        # Ensure all required keys are present
        required_keys = ['gender', 'age', 'urea', 'cr', 'hba1c', 'chol', 'tg', 'hdl', 'ldl', 'vldl', 'bmi']
        for key in required_keys:
            if key not in features:
                logger.error(f"Missing required feature: {key}")
                return "Error", 0.0, [], f"Missing required feature: {key}"
        
        # Define path for stacking model
        stacking_model_path = os.path.join(current_dir, 'models', 'stacking_model.joblib')
        
        # Check if stacking model exists, if not look in parent directory
        if not os.path.exists(stacking_model_path):
            logger.info("Stacking model not found in predictor directory, checking parent directory")
            parent_dir = os.path.dirname(current_dir)
            stacking_model_path = os.path.join(parent_dir, 'models', 'stacking_model.joblib')
            
        # Check if stacking model exists
        if not os.path.exists(stacking_model_path):
            logger.error(f"Stacking model not found at {stacking_model_path}")
            # Fall back to regular ensemble approach
            logger.info("Falling back to regular ensemble approach")
            # We'll continue with the rest of the function which loads individual models
        else:
            logger.info(f"Using stacking model from: {stacking_model_path}")
        
        # Define paths for other artifacts
        class_encoder_path = os.path.join(current_dir, 'class_encoder.joblib')
        scaler_path = os.path.join(current_dir, 'scaler.joblib')
        power_transformer_path = os.path.join(current_dir, 'power_transformer.joblib')
        gender_encoder_path = os.path.join(current_dir, 'gender_encoder.joblib')
        feature_columns_path = os.path.join(current_dir, 'feature_columns.joblib')
        
        # Check for files in parent directory if not found
        for artifact_path in [class_encoder_path, scaler_path, power_transformer_path, 
                             gender_encoder_path, feature_columns_path]:
            basename = os.path.basename(artifact_path)
            if not os.path.exists(artifact_path):
                logger.info(f"{basename} not found in predictor directory, checking parent directory")
                parent_dir = os.path.dirname(current_dir)
                new_path = os.path.join(parent_dir, basename)
                if os.path.exists(new_path):
                    # If file exists in parent directory, update the path
                    if artifact_path == class_encoder_path:
                        class_encoder_path = new_path
                    elif artifact_path == scaler_path:
                        scaler_path = new_path
                    elif artifact_path == power_transformer_path:
                        power_transformer_path = new_path
                    elif artifact_path == gender_encoder_path:
                        gender_encoder_path = new_path
                    elif artifact_path == feature_columns_path:
                        feature_columns_path = new_path
                else:
                    logger.error(f"Could not find {basename} in either predictor or parent directory")
        
        # Load class encoder or create a default one
        try:
            if os.path.exists(class_encoder_path):
                logger.info(f"Loading class encoder from: {class_encoder_path}")
                class_encoder = joblib.load(class_encoder_path)
            else:
                logger.warning(f"Class encoder not found at {class_encoder_path}, creating a default one")
                from sklearn.preprocessing import LabelEncoder
                class_encoder = LabelEncoder()
                class_encoder.fit(['No', 'Possible', 'Yes'])
                joblib.dump(class_encoder, os.path.join(current_dir, 'class_encoder.joblib'))
        except Exception as e:
            logger.error(f"Error loading class encoder: {str(e)}")
            return "Error", 0.0, [], f"Error loading class encoder: {str(e)}"
        
        # Load other artifacts or create defaults
        try:
            if os.path.exists(scaler_path):
                logger.info(f"Loading scaler from: {scaler_path}")
                scaler = joblib.load(scaler_path)
            else:
                logger.warning(f"Scaler not found at {scaler_path}, creating a default one")
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                # Since we can't fit it without data, we'll create a dummy scaler
                # This will likely produce bad results but allows the pipeline to continue
                joblib.dump(scaler, os.path.join(current_dir, 'scaler.joblib'))
            
            if os.path.exists(power_transformer_path):
                logger.info(f"Loading power transformer from: {power_transformer_path}")
                power_transformer = joblib.load(power_transformer_path)
            else:
                logger.warning(f"Power transformer not found, will not apply transformation")
                power_transformer = None
            
            if os.path.exists(feature_columns_path):
                logger.info(f"Loading feature columns from: {feature_columns_path}")
                feature_columns = joblib.load(feature_columns_path)
            else:
                logger.warning(f"Feature columns not found, will use basic features")
                feature_columns = ['Gender', 'Age', 'BMI', 'HbA1c', 'Cholesterol', 'HDL', 'Urea', 'Cr', 'TG', 'LDL', 'VLDL']
        except Exception as e:
            logger.error(f"Error loading preprocessing artifacts: {str(e)}")
            return "Error", 0.0, [], f"Error loading model artifacts: {str(e)}"
        
        # Try to encode gender or use a default value on failure
        try:
            gender_str = str(features['gender']).strip().upper()
            logger.info(f"Processing gender: {gender_str}")
            
            if os.path.exists(gender_encoder_path):
                logger.info(f"Loading gender encoder from: {gender_encoder_path}")
                gender_encoder = joblib.load(gender_encoder_path)
            else:
                logger.warning("Gender encoder not found, creating a new one")
                from sklearn.preprocessing import LabelEncoder
                gender_encoder = LabelEncoder()
                gender_encoder.fit(['M', 'F'])
                joblib.dump(gender_encoder, os.path.join(current_dir, 'gender_encoder.joblib'))
            
            try:
                gender_encoded = gender_encoder.transform([gender_str])[0]
                logger.info(f"Encoded gender {gender_str} to {gender_encoded}")
            except Exception as e:
                logger.warning(f"Error encoding gender: {str(e)}. Using default value.")
                gender_encoded = 0  # Default to the first class
        except Exception as e:
            logger.error(f"Error processing gender: {str(e)}")
            gender_encoded = 0
        
        # Handle numeric features, with defaults for missing or invalid values
        try:
            # Round all numeric values to 2 decimal places
            age = round(float(features.get('age', 45.0)), 2)
            urea = round(float(features.get('urea', 30.0)), 2)
            cr = round(float(features.get('cr', 0.9)), 2)
            hba1c = round(float(features.get('hba1c', 5.7)), 2)
            chol = round(float(features.get('chol', 200.0)), 2)
            tg = round(float(features.get('tg', 150.0)), 2)
            hdl = round(float(features.get('hdl', 50.0)), 2)
            ldl = round(float(features.get('ldl', 120.0)), 2)
            vldl = round(float(features.get('vldl', 25.0)), 2)
            bmi = round(float(features.get('bmi', 25.0)), 2)
        except Exception as e:
            logger.error(f"Error processing numeric features: {str(e)}")
            # Use default values if there's an error
            age, urea, cr, hba1c = 45.0, 30.0, 0.9, 5.7
            chol, tg, hdl, ldl, vldl, bmi = 200.0, 150.0, 50.0, 120.0, 25.0, 25.0
        
        # Create the feature dictionary
        feature_dict = {
            'Gender': gender_str,
            'Age': age,
            'Urea': urea,
            'Cr': cr,
            'HbA1c': hba1c,
            'Cholesterol': chol,
            'HDL': hdl,
            'TG': tg,
            'LDL': ldl,
            'VLDL': vldl,
            'BMI': bmi
        }
        
        # Create DataFrame with columns in the same order as training data
        if feature_columns:
            # Map our feature dict keys to the feature columns
            feature_map = {
                'Gender': 'Gender', 
                'Age': 'Age', 
                'Urea': 'Urea',
                'Cr': 'Cr', 
                'HbA1c': 'HbA1c', 
                'Cholesterol': 'Cholesterol',
                'HDL': 'HDL', 
                'TG': 'TG', 
                'LDL': 'LDL', 
                'VLDL': 'VLDL', 
                'BMI': 'BMI'
            }
            
            # Create a DataFrame with the feature columns in the right order
            df_features = pd.DataFrame({col: [feature_dict.get(feature_map.get(col, col), 0)] 
                                     for col in feature_columns})
            
            # Apply gender encoding
            if 'Gender' in df_features.columns:
                df_features['Gender'] = gender_encoded
            
            features_array = df_features.values
        else:
            # Fallback to basic structure
            features_array = np.array([
                gender_encoded, age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi
            ]).reshape(1, -1)
        
        logger.debug(f"Feature array shape: {features_array.shape}")
        
        # Apply preprocessing
        features_scaled = scaler.transform(features_array)
        
        if power_transformer is not None:
            try:
                features_transformed = power_transformer.transform(features_scaled)
            except Exception as e:
                logger.error(f"Error applying power transformer: {str(e)}")
                features_transformed = features_scaled
        else:
            features_transformed = features_scaled
        
        logger.debug(f"Transformed feature array shape: {features_transformed.shape}")
        
        # Make prediction using the stacking model if available
        if os.path.exists(stacking_model_path):
            try:
                logger.info("Making prediction using stacking model")
                stacking_model = joblib.load(stacking_model_path)
                prediction_encoded = stacking_model.predict(features_transformed)[0]
                pred_proba = stacking_model.predict_proba(features_transformed)[0]
                max_probability = pred_proba[prediction_encoded]
                
                # Decode the prediction
                prediction = class_encoder.inverse_transform([prediction_encoded])[0]
                logger.info(f"Stacking model prediction: {prediction}, Probability: {max_probability:.2%}")
                
            except Exception as e:
                logger.error(f"Error using stacking model: {str(e)}")
                # Fall back to a heuristic-based prediction
                logger.info("Falling back to heuristic-based prediction")
                # Calculate a diabetes risk score based on known risk factors
                risk_score = 0
                if hba1c >= 6.5:
                    risk_score += 3
                elif hba1c >= 5.7:
                    risk_score += 1
                
                if bmi >= 30:
                    risk_score += 2
                elif bmi >= 25:
                    risk_score += 1
                
                if age > 45:
                    risk_score += 1
                
                if chol > 240:
                    risk_score += 1
                
                if hdl < 40:
                    risk_score += 1
                
                # Determine prediction based on risk score
                if risk_score >= 4:
                    prediction = "Yes"
                    max_probability = 0.9
                elif risk_score >= 2:
                    prediction = "Possible"
                    max_probability = 0.7
                else:
                    prediction = "No"
                    max_probability = 0.8
                
                logger.info(f"Heuristic-based prediction: {prediction}, Probability: {max_probability:.2%}")
        else:
            # Fallback to simple rule-based prediction
            logger.info("Stacking model not available, using rule-based prediction")
            risk_score = 0
            if hba1c >= 6.5:
                risk_score += 3
            elif hba1c >= 5.7:
                risk_score += 1
            
            if bmi >= 30:
                risk_score += 2
            elif bmi >= 25:
                risk_score += 1
            
            if age > 45:
                risk_score += 1
            
            if chol > 240:
                risk_score += 1
            
            if hdl < 40:
                risk_score += 1
            
            # Determine prediction based on risk score
            if risk_score >= 4:
                prediction = "Yes"
                max_probability = 0.9
            elif risk_score >= 2:
                prediction = "Possible"
                max_probability = 0.7
            else:
                prediction = "No"
                max_probability = 0.8
            
            logger.info(f"Rule-based prediction: {prediction}, Probability: {max_probability:.2%}")
        
        # Calculate risk factors
        risk_factors = []
        risk_details = ""
        
        try:
            # Age risk
            if age > 45:
                risk_factors.append("Age above 45")
            
            # BMI risk
            if bmi >= 30:
                risk_factors.append("Obesity (BMI ≥ 30)")
            elif bmi >= 25:
                risk_factors.append("Overweight (BMI ≥ 25)")
            
            # HbA1c risk
            if hba1c >= 6.5:
                risk_factors.append("HbA1c ≥ 6.5% (Diabetes range)")
            elif hba1c >= 5.7:
                risk_factors.append("HbA1c 5.7-6.4% (Prediabetes range)")
            
            # Cholesterol risks
            if chol > 200:
                risk_factors.append("Elevated Total Cholesterol")
            if tg > 150:
                risk_factors.append("Elevated Triglycerides")
            if hdl < 40:
                risk_factors.append("Low HDL Cholesterol")
            if ldl > 100:
                risk_factors.append("Elevated LDL Cholesterol")
            
            # Cholesterol ratio risks
            chol_hdl_ratio = chol / max(hdl, 1)
            if chol_hdl_ratio > 5:
                risk_factors.append("High Cholesterol to HDL Ratio")
            
            # Kidney function risks
            if urea > 40:
                risk_factors.append("Elevated Blood Urea")
            if cr > 1.2:
                risk_factors.append("Elevated Creatinine")
            
            # Metabolic score risk
            metabolic_score = hba1c * 2 + bmi / 10 + chol / 50 + tg / 50 - hdl / 20
            if metabolic_score > 15:
                risk_factors.append("High Metabolic Risk Score")
            
            # Create risk details message
            if prediction == "Yes":
                if len(risk_factors) >= 3:
                    risk_details = "High risk of diabetes. Multiple significant risk factors identified."
                else:
                    risk_details = "High risk of diabetes. Some risk factors identified."
            elif prediction == "Possible":
                if len(risk_factors) >= 2:
                    risk_details = "Moderate risk of diabetes. Several risk factors present."
                else:
                    risk_details = "Moderate risk of diabetes. Some risk factors present."
            else:
                if len(risk_factors) >= 1:
                    risk_details = "Low risk of diabetes. Some risk factors present but overall risk is low."
                else:
                    risk_details = "Low risk of diabetes. Continue healthy lifestyle habits."
        except Exception as e:
            logger.error(f"Error calculating risk factors: {str(e)}")
            risk_factors = ["Error calculating risk factors"]
            risk_details = "Could not properly assess risk factors due to an error."
        
        return prediction, max_probability, risk_factors, risk_details
    
    except Exception as e:
        logger.error(f"Unhandled error predicting diabetes: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return "Error", 0.0, [], f"An error occurred during prediction: {str(e)}"

def get_model_metrics():
    """Retrieve model performance metrics."""
    try:
        metrics_path = os.path.join(current_dir, 'model_metrics.joblib')
        
        # Check if metrics file exists, if not look in parent directory
        if not os.path.exists(metrics_path):
            logger.info("Model metrics not found in predictor directory, checking parent directory")
            parent_dir = os.path.dirname(current_dir)
            metrics_path = os.path.join(parent_dir, 'model_metrics.joblib')
        
        if os.path.exists(metrics_path):
            metrics = joblib.load(metrics_path)
            
            # Ensure we have percentage representations for template rendering
            if 'accuracy_percent' not in metrics and 'accuracy' in metrics:
                metrics['accuracy_percent'] = f"{metrics['accuracy']:.2%}"
            if 'precision_percent' not in metrics and 'precision' in metrics:
                metrics['precision_percent'] = f"{metrics['precision']:.2%}"
            if 'recall_percent' not in metrics and 'recall' in metrics:
                metrics['recall_percent'] = f"{metrics['recall']:.2%}"
            if 'f1_percent' not in metrics and 'f1' in metrics:
                metrics['f1_percent'] = f"{metrics['f1']:.2%}"
            if 'roc_auc_percent' not in metrics and 'roc_auc' in metrics:
                metrics['roc_auc_percent'] = f"{metrics['roc_auc']:.2%}"
            
            # Ensure confusion matrix has the expected format for the template
            if 'confusion_matrix' in metrics and isinstance(metrics['confusion_matrix'], np.ndarray):
                cm = metrics['confusion_matrix']
                metrics['confusion_matrix'] = {
                    'matrix': cm.tolist(),
                    'true_negatives': cm[0, 0],
                    'false_positives': cm[0, 1] + (cm[0, 2] if cm.shape[1] > 2 else 0),
                    'false_negatives': cm[1, 0] + (cm[2, 0] if cm.shape[0] > 2 else 0),
                    'true_positives': cm[1, 1] + (cm[2, 2] if cm.shape[0] > 2 else 0),
                }
            
            return metrics
        else:
            logger.warning("Model metrics not found. Will use default metrics.")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'roc_auc': 0.0,
                'accuracy_percent': "0.00%",
                'precision_percent': "0.00%",
                'recall_percent': "0.00%",
                'f1_percent': "0.00%",
                'roc_auc_percent': "0.00%",
                'confusion_matrix': {
                    'matrix': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    'true_negatives': 0,
                    'false_positives': 0,
                    'false_negatives': 0,
                    'true_positives': 0,
                }
            }
    except Exception as e:
        logger.error(f"Error retrieving model metrics: {str(e)}")
        # Return default metrics in case of error
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'roc_auc': 0.0,
            'accuracy_percent': "0.00%",
            'precision_percent': "0.00%",
            'recall_percent': "0.00%",
            'f1_percent': "0.00%",
            'roc_auc_percent': "0.00%",
            'confusion_matrix': {
                'matrix': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                'true_negatives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'true_positives': 0,
            }
        }

if __name__ == "__main__":
    train_model() 