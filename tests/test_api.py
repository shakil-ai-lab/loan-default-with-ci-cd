from fastapi.testclient import TestClient
from app import app


client = TestClient(app)


sample_payload = {
    "person_age": 28,
    "person_income": 65000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5,
    "loan_intent": "EDUCATION",
    "loan_grade": "B",
    "loan_amnt": 12000,
    "loan_int_rate": 11.5,
    "loan_percent_income": 0.18,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 6
}


def test_home_route():

    response = client.get("/")

    assert response.status_code == 200


def test_prediction_api():

    response = client.post(
        "/predict",
        json=sample_payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "default_probability" in data


def test_prediction_value():

    response = client.post(
        "/predict",
        json=sample_payload
    )

    data = response.json()

    assert data["prediction"] in [0, 1]