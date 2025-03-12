import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import StackingClassifier
import joblib
import os
from sklearn.utils import shuffle

def train_model():
    # Load the dataset
    data = pd.read_csv('diabetes_dataset.csv')
    
    # Handle missing values and convert data types
    numeric_columns = ['Age', 'Pregnancies', 'BMI', 'Glucose', 'BloodPressure', 'HbA1c', 
                      'LDL', 'HDL', 'Triglycerides', 'WaistCircumference', 'HipCircumference', 'WHR']
    categorical_columns = ['FamilyHistory', 'DietType', 'Hypertension', 'MedicationUse']
    
    # Convert numeric columns
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Convert categorical columns
    for col in categorical_columns:
        data[col] = data[col].astype(int)
    
    # Fill missing values
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    data[categorical_columns] = data[categorical_columns].fillna(0)
    
    # Add significant noise to the data to prevent overfitting
    for col in numeric_columns:
        # Add random noise (5-15% of the standard deviation)
        noise_level = np.random.uniform(0.05, 0.15) * data[col].std()
        data[col] = data[col] + np.random.normal(0, noise_level, size=len(data))
    
    # Add noise to the target variable (flip some labels)
    # This simulates real-world misdiagnosis or data entry errors
    np.random.seed(42)
    flip_indices = np.random.choice(len(data), size=int(0.05 * len(data)), replace=False)
    data.loc[flip_indices, 'Outcome'] = 1 - data.loc[flip_indices, 'Outcome']
    
    # Separate features and target
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    # Shuffle the data to ensure randomness
    X, y = shuffle(X, y, random_state=42)
    
    # Split the data with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define base models with regularization to prevent overfitting
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=10, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3, subsample=0.8, random_state=42)),
        ('svc', SVC(kernel='rbf', C=0.5, gamma='auto', probability=True, random_state=42))
    ]
    
    # Define meta-model with regularization
    meta_model = LogisticRegression(C=0.5, penalty='l2', solver='liblinear', random_state=42)
    
    # Create stacking classifier with stratified k-fold
    stacking_classifier = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    
    # Train the stacking model
    stacking_classifier.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = stacking_classifier.predict(X_test_scaled)
    
    # Print model performance
    print("Model Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Perform cross-validation
    cv_scores = cross_val_score(stacking_classifier, X_train_scaled, y_train, 
                               cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    print("\nCross-validation scores:", cv_scores)
    print("Average CV score:", cv_scores.mean())
    
    # Create models directory if it doesn't exist
    os.makedirs('predictor/models', exist_ok=True)
    
    # Save the model and scaler
    joblib.dump(stacking_classifier, 'predictor/models/stacking_model.joblib')
    joblib.dump(scaler, 'predictor/models/scaler.joblib')
    
    # Save additional data for visualizations
    model_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_names': X.columns.tolist()
    }
    joblib.dump(model_data, 'predictor/models/model_data.joblib')
    
    return stacking_classifier, scaler

def predict_diabetes(features):
    # Load the saved model and scaler
    model = joblib.load('predictor/models/stacking_model.joblib')
    scaler = joblib.load('predictor/models/scaler.joblib')
    
    # Scale the features
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)
    
    return prediction[0], probability[0]

def get_model_metrics():
    # Load model data
    model = joblib.load('predictor/models/stacking_model.joblib')
    model_data = joblib.load('predictor/models/model_data.joblib')
    
    X_train = model_data['X_train']
    X_test = model_data['X_test']
    y_test = model_data['y_test']
    y_pred = model_data['y_pred']
    feature_names = model_data['feature_names']
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Get feature importance (from Random Forest, which is one of the base models)
    rf_model = model.estimators_[0][1]  # Get the Random Forest model
    importances = rf_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Calculate correlation matrix
    correlation_matrix = X_train.corr()
    
    # Calculate ROC curve
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Compile metrics
    metrics = {
        'accuracy': round(accuracy * 100, 2),
        'precision': round(conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1]) * 100 if (conf_matrix[1, 1] + conf_matrix[0, 1]) > 0 else 0, 2),
        'recall': round(conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]) * 100 if (conf_matrix[1, 1] + conf_matrix[1, 0]) > 0 else 0, 2),
        'f1_score': round(2 * (conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])) * (conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])) / 
                         ((conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])) + (conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]))) * 100 
                         if ((conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])) + (conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]))) > 0 else 0, 2),
        'roc_auc': round(roc_auc * 100, 2)
    }
    
    return metrics, feature_importance, correlation_matrix, conf_matrix, (fpr, tpr, roc_auc)

if __name__ == "__main__":
    train_model() 