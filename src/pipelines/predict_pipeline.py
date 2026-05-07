import pandas as pd
import logging
import yaml
import joblib
from pathlib import Path
import mlflow

# ---------------- LOGGER ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------- UTILS ---------------- #
def read_params(config_path="params.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------- LOAD ARTIFACTS ---------------- #
def load_encoder(encoder_path):
    try:
        logging.info("Loading encoder")
        return joblib.load(encoder_path)
    except Exception as e:
        logging.error(f"Error loading encoder: {e}")
        raise


def load_scaler(scaler_path):
    try:
        logging.info("Loading scaler")
        return joblib.load(scaler_path)
    except Exception as e:
        logging.error(f"Error loading scaler: {e}")
        raise


def load_model_from_registry(model_name, stage="Staging"):
    try:
        logging.info(f"Loading model from MLflow Registry: {model_name} ({stage})")

        tracking_dir = Path("mlruns").resolve().as_uri()
        mlflow.set_tracking_uri(tracking_dir)

        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)

        logging.info("Model loaded successfully")
        return model

    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


# ---------------- PREPROCESS INPUT ---------------- #
def preprocess_input(input_data, encoder, scaler):
    try:
        logging.info("Preprocessing input data")

        df = pd.DataFrame([input_data])

        # Apply encoder
        encoded = encoder.transform(df)

        # Apply scaler
        scaled = scaler.transform(encoded)

        return scaled

    except Exception as e:
        logging.error(f"Error preprocessing input: {e}")
        raise


# ---------------- PREDICT ---------------- #
def predict(input_data, config_path="params.yaml"):
    try:
        config = read_params(config_path)

        encoder_path = config["data_preprocessing"]["encoder_path"]
        scaler_path = config["model_training"]["scaler_path"]

        model_name = "LoanDefaultModel"

        encoder = load_encoder(encoder_path)
        scaler = load_scaler(scaler_path)
        model = load_model_from_registry(model_name)

        processed_input = preprocess_input(input_data, encoder, scaler)

        prediction = model.predict(processed_input)

        return int(prediction[0])

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise


# ---------------- MAIN TEST ---------------- #
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
        "LoanPurpose": "Personal",
        "HasCoSigner": "No"
    }

    result = predict(sample_input)
    print("Prediction:", result)