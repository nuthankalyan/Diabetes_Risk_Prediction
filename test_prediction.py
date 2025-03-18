import logging
from predictor.ml_model import predict_diabetes, get_model_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_prediction():
    # Sample data for prediction (Age, Gender, Pregnancies, BMI, Glucose, BloodPressure, HbA1c, LDL, HDL, 
    # Triglycerides, WaistCircumference, HipCircumference, WHR, FamilyHistory, DietType, Hypertension, MedicationUse)
    test_features_female = [45, 'Female', 2, 28.5, 140, 90, 6.1, 120, 45, 160, 95, 105, 0.91, 1, 0, 1, 1]
    test_features_male = [50, 'Male', 0, 30.2, 150, 95, 6.5, 130, 40, 180, 100, 95, 1.05, 1, 0, 1, 1]
    
    logger.info("Making prediction for female test case...")
    prediction_female, probability_female = predict_diabetes(test_features_female)
    
    logger.info("Making prediction for male test case...")
    prediction_male, probability_male = predict_diabetes(test_features_male)
    
    logger.info(f"Female test case: Prediction={prediction_female}, Probability={probability_female}")
    logger.info(f"Male test case: Prediction={prediction_male}, Probability={probability_male}")
    
    # Get model metrics
    logger.info("Getting model metrics...")
    metrics, feature_importance, _, _, _ = get_model_metrics()
    
    logger.info(f"Model metrics: {metrics}")
    logger.info(f"Top 5 important features: {feature_importance.sort_values('importance', ascending=False).head(5)}")
    
    return True

if __name__ == "__main__":
    logger.info("Starting prediction test with CatBoost model...")
    success = test_prediction()
    if success:
        logger.info("Prediction test completed successfully!")
    else:
        logger.error("Prediction test failed!") 