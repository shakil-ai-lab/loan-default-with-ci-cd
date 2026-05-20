import pytest
from src.pipelines.predict_pipeline import predict


sample_input = {
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


def test_prediction_output_type():

    result = predict(sample_input)

    assert isinstance(result, dict)


def test_prediction_contains_keys():

    result = predict(sample_input)

    assert "prediction" in result
    assert "default_probability" in result


def test_prediction_value():

    result = predict(sample_input)

    assert result["prediction"] in [0, 1]


def test_probability_range():

    result = predict(sample_input)

    assert 0.0 <= result["default_probability"] <= 1.0