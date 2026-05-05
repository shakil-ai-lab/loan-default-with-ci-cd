import os
import pandas as pd
import yaml
import logging
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from imblearn.over_sampling import ADASYN

# ---------------- LOGGER ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------- UTILS ---------------- #
def read_params(config_path="params.yaml"):
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error reading params.yaml: {e}")
        raise


# ---------------- CORE FUNCTIONS ---------------- #

def load_data(path):
    try:
        logging.info(f"Loading processed data from {path}")
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def split_data(df, test_size, random_state):
    try:
        logging.info("Splitting data into train and test")

        X = df.drop("Default", axis=1)
        y = df["Default"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error in splitting data: {e}")
        raise


def apply_adasyn(X_train, y_train):
    try:
        logging.info("Applying ADASYN for imbalance handling")

        adasyn = ADASYN(random_state=42)
        X_res, y_res = adasyn.fit_resample(X_train, y_train)

        logging.info(f"Before ADASYN: {y_train.value_counts().to_dict()}")
        logging.info(f"After ADASYN: {pd.Series(y_res).value_counts().to_dict()}")

        return X_res, y_res

    except Exception as e:
        logging.error(f"Error applying ADASYN: {e}")
        raise


def apply_scaling(X_train, X_test):
    try:
        logging.info("Applying StandardScaler")

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, scaler

    except Exception as e:
        logging.error(f"Error in scaling: {e}")
        raise


def train_model(X_train, y_train, params):
    try:
        logging.info("Training Logistic Regression model")

        model = LogisticRegression(
            C=params["C"],
            penalty=params["penalty"],
            solver=params["solver"],
            max_iter=params["max_iter"]
        )

        model.fit(X_train, y_train)

        return model

    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise


def evaluate_model(model, X_test, y_test):
    try:
        logging.info("Evaluating model")

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }

        logging.info(f"Metrics: {metrics}")

        return metrics

    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise


def save_artifacts(model, scaler, model_path, scaler_path):
    try:
        logging.info("Saving model and scaler")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        logging.info("Artifacts saved")

    except Exception as e:
        logging.error(f"Error saving artifacts: {e}")
        raise


# ---------------- PIPELINE ---------------- #
def run_training(config_path="params.yaml"):
    try:
        config = read_params(config_path)["model_training"]

        # MLflow setup
        mlflow.set_experiment("Loan_Default_Prediction")

        with mlflow.start_run():

            df = load_data(config["input_data_path"])

            X_train, X_test, y_train, y_test = split_data(
                df,
                config["test_size"],
                config["random_state"]
            )

            # ADASYN
            X_res, y_res = apply_adasyn(X_train, y_train)

            # Scaling AFTER ADASYN
            X_train_scaled, X_test_scaled, scaler = apply_scaling(X_res, X_test)

            # Train
            model = train_model(X_train_scaled, y_res, config["model_params"])

            # Evaluate
            metrics = evaluate_model(model, X_test_scaled, y_test)

            # Log params
            mlflow.log_params(config["model_params"])

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Save locally
            save_artifacts(
                model,
                scaler,
                config["model_path"],
                config["scaler_path"]
            )

            logging.info("Training pipeline completed")

    except Exception as e:
        logging.error(f"Training pipeline failed: {e}")
        raise


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    run_training()