from predictor.ml_model import train_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting model training with CatBoost and other algorithms...")
    success = train_model()
    if success:
        logger.info("Model training completed successfully!")
    else:
        logger.error("Model training failed!") 