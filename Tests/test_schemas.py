from datetime import date
import pytest
from pydantic import ValidationError
from fantasy_basketball.schemas import PredictionRequest, PredictionResponse

@pytest.fixture
def prediction_payload():
    return {"player_id": 203999, "season": "2025-26", "game_date": "2026-02-05"}

def test_prediction_request_parses_dates_and_sets_defaults(prediction_payload):
    request = PredictionRequest(**prediction_payload)
    assert request.player_id == 203999
    assert request.season == "2025-26"
    assert request.game_date == date(2026, 2, 5)
    assert request.model == "linear_regression"
    assert request.as_of_date is None

@pytest.mark.parametrize("model", ["linear_regression", "random_forest", "pytorch"])
@pytest.mark.parametrize("cutoff", ["2026-02-01", "2026-02-05"])
def test_prediction_request_accepts_models_and_valid_cutoffs(prediction_payload, model, cutoff):
    request = PredictionRequest(**{**prediction_payload, "model": model, "as_of_date": cutoff})
    assert request.model == model
    assert request.as_of_date == date.fromisoformat(cutoff)

@pytest.mark.parametrize("field", ["player_id", "season", "game_date"])
def test_prediction_request_rejects_missing_required_field(prediction_payload, field):
    del prediction_payload[field]
    with pytest.raises(ValidationError) as error:
        PredictionRequest(**prediction_payload)
    assert any(item["loc"] == (field,) and item["type"] == "missing" for item in error.value.errors())

@pytest.mark.parametrize("field, value", [
    ("player_id", 0),
    ("player_id", -1),
    ("player_id", "not-an-id"),
    ("season", "2025"),
    ("season", "25-26"),
    ("season", "2025-2026"),
    ("game_date", "not-a-date"),
    ("game_date", "2026-02-30"),
    ("model", "unknown"),
    ("as_of_date", "not-a-date")
])
def test_prediction_request_rejects_invalid_field(prediction_payload, field, value):
    with pytest.raises(ValidationError) as error:
        PredictionRequest(**{**prediction_payload, field: value})
    assert any(item["loc"] == (field,) for item in error.value.errors())

def test_prediction_request_rejects_cutoff_after_game(prediction_payload):
    with pytest.raises(ValidationError, match="as_of_date cannot be after game_date"):
        PredictionRequest(**{**prediction_payload, "as_of_date": "2026-02-06"})

def test_prediction_request_serializes_dates_for_json(prediction_payload):
    payload = {**prediction_payload, "model": "pytorch", "as_of_date": "2026-02-01"}
    request = PredictionRequest(**payload)
    assert request.model_dump(mode="json") == payload

@pytest.fixture
def prediction_response_payload():
    return {
        "player_id": 203999,
        "season": "2025-26",
        "game_date": "2026-02-05",
        "model": "linear_regression",
        "as_of_date": "2026-02-01",
        "projected_fp": 57.58}

def test_prediction_response_parses_and_serializes(prediction_response_payload):
    response = PredictionResponse(**prediction_response_payload)
    assert response.game_date == date(2026, 2, 5)
    assert response.as_of_date == date(2026, 2, 1)
    assert response.projected_fp == pytest.approx(57.58)
    assert response.model_dump(mode="json") == prediction_response_payload

@pytest.mark.parametrize("model", ["linear_regression", "random_forest", "pytorch"])
def test_prediction_response_accepts_supported_models(prediction_response_payload, model):
    response = PredictionResponse(**{**prediction_response_payload, "model": model})
    assert response.model == model

@pytest.mark.parametrize("points", [0.0, -5.0, 42.5])
def test_prediction_response_accepts_finite_points(prediction_response_payload, points):
    response = PredictionResponse(**{**prediction_response_payload, "projected_fp": points})
    assert response.projected_fp == points

@pytest.mark.parametrize("points", [float("nan"), float("inf"), float("-inf")])
def test_prediction_response_rejects_nonfinite_points(prediction_response_payload, points):
    with pytest.raises(ValidationError) as error:
        PredictionResponse(**{**prediction_response_payload, "projected_fp": points})
    assert any(item["loc"] == ("projected_fp",) for item in error.value.errors())

@pytest.mark.parametrize("field", ["player_id", "season", "game_date", "model", "as_of_date", "projected_fp"])
def test_prediction_response_requires_all_fields(prediction_response_payload, field):
    del prediction_response_payload[field]
    with pytest.raises(ValidationError) as error:
        PredictionResponse(**prediction_response_payload)
    assert any(item["loc"] == (field,) and item["type"] == "missing" for item in error.value.errors())

@pytest.mark.parametrize("field, value", [
    ("player_id", 0),
    ("season", "2025"),
    ("game_date", "not-a-date"),
    ("model", "unknown"),
    ("as_of_date", None),
    ("projected_fp", "not-a-number")
])
def test_prediction_response_rejects_invalid_fields(prediction_response_payload, field, value):
    with pytest.raises(ValidationError) as error:
        PredictionResponse(**{**prediction_response_payload, field: value})
    assert any(item["loc"] == (field,) for item in error.value.errors())