import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def create_features(df):
    """
    Create additional features to improve model performance
    """
    # Make a copy to avoid modifying the original dataframe
    df_new = df.copy()
    
    # Create interaction features between key health indicators
    logging.info("Creating interaction features")
    
    # BMI and Age interaction (age-related BMI risk)
    df_new['BMI_Age_Ratio'] = df_new['BMI'] / df_new['Age']
    
    # HbA1c and BMI interaction (key diabetes indicators together)
    df_new['HbA1c_BMI'] = df_new['HbA1c'] * df_new['BMI']
    
    # Cholesterol ratios (important cardiovascular and metabolic risk indicators)
    df_new['Chol_HDL_Ratio'] = df_new['Chol'] / df_new['HDL']
    df_new['LDL_HDL_Ratio'] = df_new['LDL'] / df_new['HDL']
    df_new['TG_HDL_Ratio'] = df_new['TG'] / df_new['HDL']
    
    # Create metabolic risk score (combines multiple risk factors)
    df_new['Metabolic_Score'] = (
        df_new['HbA1c'] + 
        df_new['BMI'] / 10 + 
        df_new['Chol'] / 50 + 
        df_new['TG'] / 50 - 
        df_new['HDL'] / 10
    )
    
    # Create kidney function score
    df_new['Kidney_Score'] = df_new['Urea'] * df_new['Cr']
    
    # Create age groups (categorical risk bands)
    df_new['Age_Group'] = pd.cut(
        df_new['Age'], 
        bins=[0, 30, 45, 60, 100], 
        labels=[0, 1, 2, 3]
    ).astype(int)
    
    # Round all features to 2 decimal places
    numeric_cols = df_new.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'CLASS' and col != 'Age_Group':  # Don't round categorical values
            df_new[col] = df_new[col].round(2)
    
    return df_new

def train_and_evaluate_model(dataset_path):
    """
    Train and evaluate an enhanced diabetes prediction model with improved accuracy
    """
    try:
        logging.info(f"Loading dataset from {dataset_path}")
        if os.path.exists(dataset_path):
            # Load the dataset
            df = pd.read_csv(dataset_path)
        else:
            logging.error(f"Dataset not found at {dataset_path}. Please create the dataset first.")
            return None
        
        # Print dataset info
        logging.info(f"Dataset shape: {df.shape}")
        logging.info(f"Dataset columns: {df.columns.tolist()}")
        
        # Drop non-predictive columns like ID if they exist
        if 'ID' in df.columns:
            df = df.drop('ID', axis=1)
        
        # Round numeric columns to 2 decimal places
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'CLASS':  # Don't round categorical target
                df[col] = df[col].round(2)
                logging.info(f"Rounded {col} to 2 decimal places")
        
        # Check for NaN values
        logging.info(f"NaN values in dataset: {df.isna().sum().sum()}")
        
        # Create new features
        logging.info("Enhancing features for better prediction")
        df = create_features(df)
        
        # Handle categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        logging.info(f"Categorical columns: {categorical_cols}")
        
        # Process Gender column separately
        gender_encoder = LabelEncoder()
        if 'Gender' in df.columns:
            # Make sure we handle the gender column correctly
            logging.info("Encoding Gender column")
            # First, clean up any whitespace or case issues
            df['Gender'] = df['Gender'].str.strip().str.upper()
            # Then encode it
            df['Gender_Encoded'] = gender_encoder.fit_transform(df['Gender'])
            # Save the encoder
            joblib.dump(gender_encoder, 'gender_encoder.joblib')
            # Drop the original Gender column
            df = df.drop('Gender', axis=1)
        
        # Encode other categorical columns if any exist
        encoders = {}
        for col in categorical_cols:
            if col != 'Gender' and col != 'CLASS':  # Skip Gender and target
                logging.info(f"Encoding {col}")
                encoder = LabelEncoder()
                df[f'{col}_Encoded'] = encoder.fit_transform(df[col])
                encoders[col] = encoder
                df = df.drop(col, axis=1)
        
        # Fill missing values (for numerical columns)
        logging.info("Handling missing values")
        # Use only numeric columns for filling with mean
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            df[col] = df[col].fillna(round(df[col].mean(), 2))
        
        # Extract features and target
        X = df.drop('CLASS', axis=1)
        y = df['CLASS']
        
        # Save feature columns for prediction
        feature_cols = X.columns.tolist()
        joblib.dump(feature_cols, 'feature_columns.joblib')
        
        # Encode the target
        class_encoder = LabelEncoder()
        y_encoded = class_encoder.fit_transform(y)
        # Save the class encoder
        joblib.dump(class_encoder, 'class_encoder.joblib')
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Save the scaler
        joblib.dump(scaler, 'scaler.joblib')
        
        # Apply SMOTE to balance the classes (extremely important for imbalanced data)
        logging.info("Applying SMOTE to balance classes")
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
        
        # Use feature selection to identify most important features
        logging.info("Performing feature selection")
        selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42),
            threshold="median"
        )
        selector.fit(X_train_smote, y_train_smote)
        
        # Get selected features
        X_train_selected = selector.transform(X_train_smote)
        X_test_selected = selector.transform(X_test_scaled)
        
        # Save the selector
        joblib.dump(selector, 'feature_selector.joblib')
        
        # Define optimized base models
        logging.info("Training optimized ensemble model")
        rf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=42
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
        
        svm = SVC(
            C=10,
            kernel='rbf',
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
        
        knn = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            p=2  # Euclidean distance
        )
        
        mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42
        )
        
        # Train base models
        logging.info("Training Random Forest model")
        rf.fit(X_train_selected, y_train_smote)
        
        logging.info("Training Gradient Boosting model")
        gb.fit(X_train_selected, y_train_smote)
        
        logging.info("Training SVM model")
        svm.fit(X_train_selected, y_train_smote)
        
        logging.info("Training KNN model")
        knn.fit(X_train_selected, y_train_smote)
        
        logging.info("Training Neural Network model")
        mlp.fit(X_train_selected, y_train_smote)
        
        # Create an improved voting classifier
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('svm', svm),
                ('knn', knn),
                ('mlp', mlp)
            ],
            voting='soft'  # Use probability estimates for voting
        )
        
        logging.info("Training Voting Classifier")
        voting_clf.fit(X_train_selected, y_train_smote)
        
        # Make predictions with voting classifier
        y_pred = voting_clf.predict(X_test_selected)
        y_pred_proba = voting_clf.predict_proba(X_test_selected)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Calculate ROC AUC for multi-class
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        except Exception as e:
            logging.warning(f"Could not calculate ROC AUC: {e}")
            roc_auc = None
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Generate classification report
        cr = classification_report(y_test, y_pred, target_names=class_encoder.classes_)
        
        # Log metrics
        logging.info(f"\nAccuracy: {accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")
        if roc_auc:
            logging.info(f"ROC AUC: {roc_auc:.4f}")
        
        # Log confusion matrix
        logging.info("\nConfusion Matrix:")
        logging.info(cm)
        
        # Log classification report
        logging.info("\nClassification Report:")
        logging.info(cr)
        
        # Save the model and metrics
        joblib.dump(voting_clf, 'diabetes_model.joblib')
        
        # Save metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': cr
        }
        
        joblib.dump(metrics, 'model_metrics.joblib')
        
        # Return the accuracy
        return accuracy
    
    except Exception as e:
        logging.error(f"Error training and evaluating model: {e}")
        return None

if __name__ == "__main__":
    # Path to the dataset
    dataset_path = os.path.join(os.getcwd(), "Balanced_Diabetes_Dataset.csv")
    
    # Train and evaluate the model
    accuracy = train_and_evaluate_model(dataset_path)
    
    if accuracy:
        logging.info(f"\nModel training and evaluation completed successfully.")
        logging.info(f"Model accuracy: {accuracy:.4f}")
    else:
        logging.error("Model training and evaluation failed.") 