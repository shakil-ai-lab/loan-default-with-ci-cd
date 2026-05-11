from fastapi import FastAPI
from src.pipelines.predict_pipeline import predict

from src.schemas.prediction_schema import (
    LoanInput,
    PredictionResponse
)

app = FastAPI(
    title="Loan Default Prediction API",
    version="1.0"
)


@app.get("/")
def home():
    return {
        "message": "API is running"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def prediction(data: LoanInput):

    input_data = data.model_dump()

    result = predict(input_data)

    response = PredictionResponse(
        prediction=int(result["prediction"]),
        default_probability=result["default_probability"]
    )

    return response