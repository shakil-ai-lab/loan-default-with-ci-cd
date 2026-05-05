import os
import pandas as pd
import yaml
import logging

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
        logging.info(f"Loading data from {path}")
        df = pd.read_csv(path)
        logging.info(f"Data loaded successfully with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def basic_cleaning(df):
    try:
        logging.info("Starting basic cleaning")

        # Remove ID column
        df = df.drop("LoanID", axis=1, errors="ignore")

        # Clean binary/ordinal columns
        for col in ['Education', 'HasCoSigner', 'HasMortgage', 'HasDependents']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        logging.info("Basic cleaning completed")
        return df

    except Exception as e:
        logging.error(f"Error in cleaning: {e}")
        raise


def save_data(df, output_path):
    try:
        logging.info(f"Saving cleaned data to {output_path}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        logging.info("Data saved successfully")

    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise


# ---------------- PIPELINE FUNCTION ---------------- #
def run_data_ingestion(config_path="params.yaml"):
    try:
        config = read_params(config_path)["data_ingestion"]

        df = load_data(config["raw_data_path"])
        df = basic_cleaning(df)

        save_data(df, config["clean_data_path"])

        logging.info("Data ingestion stage completed successfully")

    except Exception as e:
        logging.error(f"Data ingestion pipeline failed: {e}")
        raise


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    run_data_ingestion()