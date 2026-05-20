import os
import joblib
import pytest


MODEL_PATH = "models/model.pkl"
SCALER_PATH = "src/artifacts/scaler.pkl"
ENCODER_PATH = "src/artifacts/encoder.pkl"


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), \
        f"Model file not found at {MODEL_PATH}"


def test_scaler_file_exists():
    assert os.path.exists(SCALER_PATH), \
        f"Scaler file not found at {SCALER_PATH}"


def test_encoder_file_exists():
    assert os.path.exists(ENCODER_PATH), \
        f"Encoder file not found at {ENCODER_PATH}"


def test_model_loading():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_scaler_loading():
    scaler = joblib.load(SCALER_PATH)
    assert scaler is not None


def test_encoder_loading():
    encoder = joblib.load(ENCODER_PATH)
    assert encoder is not None