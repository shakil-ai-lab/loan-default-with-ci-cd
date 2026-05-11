import os
import logging
from pathlib import Path

import dagshub
import joblib
import mlflow
import pandas as pd
import yaml


# =========================================================
# LOGGER CONFIG
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# BASE DIRECTORY (Docker + Windows + Linux Safe)
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent


# =========================================================
# READ PARAMS
# =========================================================
def read_params(config_path="params.yaml"):
    """
    Read YAML configuration file
    """
    try:
        config_full_path = os.path.join(BASE_DIR, config_path)

        with open(config_full_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)

        logger.info("params.yaml loaded successfully")

        return config

    except Exception as e:
        logger.error(f"Error reading params.yaml: {e}")
        raise


# =========================================================
# LOAD LOCAL ARTIFACTS
# =========================================================
def load_local_artifact(relative_path):
    """
    Load local artifacts from Docker container / local filesystem
    """

    try:
        full_path = os.path.join(BASE_DIR, relative_path)

        logger.info(f"Loading local artifact from: {full_path}")

        artifact = joblib.load(full_path)

        logger.info("Artifact loaded successfully")

        return artifact

    except Exception as e:
        logger.error(f"Error loading artifact: {e}")
        raise


# =========================================================
# DAGSHUB + MLFLOW SETUP
# =========================================================
def setup_mlflow():
    """
    Configure DagsHub MLflow tracking
    """

    try:
        dagshub.init(
            repo_owner="shakil-ai-lab",
            repo_name="loan-default-with-ci-cd",
            mlflow=True
        )

        mlflow.set_tracking_uri(
            "https://dagshub.com/shakil-ai-lab/loan-default-with-ci-cd.mlflow"
        )

        logger.info("DagsHub + MLflow configured successfully")

    except Exception as e:
        logger.error(f"Error configuring MLflow: {e}")
        raise


# =========================================================
# LOAD MODEL FROM REGISTRY
# =========================================================
def load_model_from_registry(
    model_name="LoanDefaultModel",
    stage="Staging"
):
    """
    Load model dynamically from MLflow Registry
    """

    try:
        logger.info(
            f"Loading model from registry: {model_name} ({stage})"
        )

        model_uri = f"models:/{model_name}/{stage}"

        model = mlflow.sklearn.load_model(model_uri)

        logger.info("Model loaded successfully from registry")

        return model

    except Exception as e:
        logger.error(f"Error loading model from registry: {e}")
        raise


# =========================================================
# LOAD ALL OBJECTS ONCE (IMPORTANT FOR FASTAPI PERFORMANCE)
# =========================================================
try:

    logger.info("Initializing prediction pipeline objects")

    config = read_params()

    # Setup MLflow
    setup_mlflow()

    # Load encoder
    encoder = load_local_artifact(
        config["data_preprocessing"]["encoder_path"]
    )

    # Load scaler
    scaler = load_local_artifact(
        config["model_training"]["scaler_path"]
    )

    # Load model
    model = load_model_from_registry(
        model_name="LoanDefaultModel",
        stage="Staging"
    )

    logger.info("All prediction objects initialized successfully")

except Exception as e:
    logger.error(f"Pipeline initialization failed: {e}")
    raise


# =========================================================
# PREPROCESS INPUT
# =========================================================
def preprocess_input(input_data):
    """
    Preprocess incoming input data
    """

    try:
        logger.info("Starting input preprocessing")

        # Convert input dict to dataframe
        df = pd.DataFrame([input_data])

        logger.info("Input converted to DataFrame")

        # Apply encoder
        encoded_data = encoder.transform(df)

        logger.info("Encoding completed")

        # Apply scaler
        scaled_data = scaler.transform(encoded_data)

        logger.info("Scaling completed")

        return scaled_data

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


# =========================================================
# PREDICT FUNCTION
# =========================================================
def predict(input_data):
    """
    Predict loan default
    """

    try:
        logger.info("Prediction request received")

        # Preprocess
        processed_input = preprocess_input(input_data)

        # Prediction
        prediction = model.predict(processed_input)

        # Probability
        probability = model.predict_proba(processed_input)

        result = {
            "prediction": int(prediction[0]),
            "default_probability": round(float(probability[0][1]), 4)
        }

        logger.info(f"Prediction successful: {result}")

        return result

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


# =========================================================
# MAIN TEST
# =========================================================
if __name__ == "__main__":

    sample_input = {
        "Age": 35,
        "Income": 60000,
        "LoanAmount": 20000,
        "CreditScore": 650,
        "MonthsEmployed": 24,
        "NumCreditLines": 3,
        "InterestRate": 12.5,
        "LoanTerm": 36,
        "DTIRatio": 0.3,
        "Education": "Bachelor's",
        "EmploymentType": "Full-time",
        "MaritalStatus": "Single",
        "HasMortgage": "No",
        "HasDependents": "Yes",
        "LoanPurpose": "Home",
        "HasCoSigner": "No"
    }

    prediction_result = predict(sample_input)

    print(prediction_result)