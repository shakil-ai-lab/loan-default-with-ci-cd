import os
import pandas as pd
import yaml
import logging
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# ---------------- LOGGER ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------- UTILS ---------------- #
def read_params(config_path="params.yaml"):
    try:
        with open(config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config
    except Exception as e:
        logging.error(f"Error reading params.yaml: {e}")
        raise


# ---------------- CORE FUNCTIONS ---------------- #

def load_data(path):
    try:
        logging.info(f"Loading cleaned data from {path}")
        df = pd.read_csv(path)
        logging.info(f"Data shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def define_features():
    try:
        logging.info("Defining feature groups")

        binary_features = ['HasCoSigner', 'HasMortgage', 'HasDependents']
        binary_order = [["No", "Yes"]] * len(binary_features)

        numeric_features = [
            'Age', 'Income', 'LoanAmount', 'CreditScore',
            'MonthsEmployed', 'NumCreditLines', 'InterestRate',
            'LoanTerm', 'DTIRatio'
        ]

        categorical_features = ['EmploymentType', 'MaritalStatus', 'LoanPurpose']

        ordinal_feature = ['Education']
        education_order = [["High School", "Bachelor's", "Master's", "PhD"]]

        return {
            "binary_features": binary_features,
            "binary_order": binary_order,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "ordinal_feature": ordinal_feature,
            "education_order": education_order
        }

    except Exception as e:
        logging.error(f"Error defining features: {e}")
        raise


def create_encoder(features):
    try:
        logging.info("Creating ColumnTransformer encoder")

        encoder = ColumnTransformer([
            ("num", "passthrough", features["numeric_features"]),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), features["categorical_features"]),
            ("bin", OrdinalEncoder(categories=features["binary_order"]), features["binary_features"]),
            ("ord", OrdinalEncoder(categories=features["education_order"]), features["ordinal_feature"]),
        ])

        return encoder

    except Exception as e:
        logging.error(f"Error creating encoder: {e}")
        raise


def apply_encoding(df, encoder):
    try:
        logging.info("Applying encoding")

        X = df.drop("Default", axis=1)
        y = df["Default"]

        X_encoded = encoder.fit_transform(X)

        # Convert to DataFrame (important for traceability)
        X_encoded_df = pd.DataFrame(X_encoded)

        processed_df = pd.concat([X_encoded_df, y.reset_index(drop=True)], axis=1)

        logging.info("Encoding completed")

        return processed_df, encoder

    except Exception as e:
        logging.error(f"Error in encoding: {e}")
        raise


def save_outputs(df, encoder, data_path, encoder_path):
    try:
        logging.info("Saving processed data and encoder")

        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)

        df.to_csv(data_path, index=False)
        joblib.dump(encoder, encoder_path)

        logging.info("Artifacts saved successfully")

    except Exception as e:
        logging.error(f"Error saving outputs: {e}")
        raise


# ---------------- PIPELINE FUNCTION ---------------- #
def run_data_preprocessing(config_path="params.yaml"):
    try:
        config = read_params(config_path)["data_preprocessing"]

        df = load_data(config["input_data_path"])
        features = define_features()

        encoder = create_encoder(features)
        processed_df, encoder = apply_encoding(df, encoder)

        save_outputs(
            processed_df,
            encoder,
            config["processed_data_path"],
            config["encoder_path"]
        )

        logging.info("Data preprocessing stage completed")

    except Exception as e:
        logging.error(f"Preprocessing pipeline failed: {e}")
        raise


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    run_data_preprocessing()