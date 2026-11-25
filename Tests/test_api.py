from fastapi.testclient import TestClient
from fantasy_basketball.api import app
from unittest.mock import Mock
import pytest
from fantasy_basketball import api as api_module

def test_health_endpoint_returns_ok():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.fixture
def prediction_api(monkeypatch, tmp_path):
    db_path = tmp_path / "test.db"
    db_path.touch()
    configs = {}
    for name, filename, model_type in [
        ("linear_regression", "linear.pkl", "sklearn"),
        ("random_forest", "forest.pkl", "sklearn"),
        ("pytorch", "network.pt", "pytorch")]:
        path = tmp_path / filename
        path.touch()
        configs[name] = (path, model_type)
    connection = Mock()
    connection.execute.return_value.fetchone.return_value = (203999,)
    connector = Mock(return_value=connection)
    predictor = Mock(return_value=57.58)
    monkeypatch.setattr(api_module, "DB_PATH", db_path)
    monkeypatch.setattr(api_module, "MODEL_CONFIGS", configs)
    monkeypatch.setattr(api_module, "get_connection", connector)
    monkeypatch.setattr(api_module, "predict_stored_player_next_game", predictor)
    with TestClient(api_module.app) as client:
        yield {
            "client": client,
            "connection": connection,
            "connector": connector,
            "predictor": predictor,
            "db_path": db_path,
            "configs": configs}

@pytest.fixture
def prediction_body():
    return {"player_id": 203999, "season": "2025-26", "game_date": "2026-02-05"}

@pytest.mark.parametrize("model", ["linear_regression", "random_forest", "pytorch"])
@pytest.mark.parametrize("cutoff", [None, "2026-02-01"])
def test_predict_endpoint_selects_model_and_returns_projection(prediction_api, prediction_body, model, cutoff):
    source = prediction_api
    payload = {**prediction_body, "model": model}
    if cutoff is not None:
        payload["as_of_date"] = cutoff
    response = source["client"].post("/predict", json=payload)
    effective_cutoff = cutoff or prediction_body["game_date"]
    assert response.status_code == 200
    assert response.json() == {
        **prediction_body,
        "model": model,
        "as_of_date": effective_cutoff,
        "projected_fp": 57.58}
    model_path, model_type = source["configs"][model]
    source["connector"].assert_called_once_with(str(source["db_path"]))
    source["predictor"].assert_called_once_with(
        source["connection"],
        203999,
        "2025-26",
        "2026-02-05",
        str(model_path),
        model_type,
        as_of_date=effective_cutoff)
    query, parameters = source["connection"].execute.call_args.args
    assert "WHERE player_id = ?" in query
    assert parameters == (203999,)
    source["connection"].close.assert_called_once_with()

def test_predict_endpoint_defaults_to_linear_regression(prediction_api, prediction_body):
    source = prediction_api
    response = source["client"].post("/predict", json=prediction_body)
    assert response.status_code == 200
    assert response.json()["model"] == "linear_regression"
    assert response.json()["as_of_date"] == prediction_body["game_date"]
    expected_path, expected_type = source["configs"]["linear_regression"]
    assert source["predictor"].call_args.args[4:6] == (str(expected_path), expected_type)

@pytest.mark.parametrize("field, value", [
    ("player_id", 0),
    ("season", "2025"),
    ("game_date", "2026-02-30"),
    ("model", "unknown"),
    ("as_of_date", "2026-02-06")
])
def test_predict_endpoint_rejects_invalid_requests_before_database_access(prediction_api, prediction_body, field, value):
    source = prediction_api
    response = source["client"].post("/predict", json={**prediction_body, field: value})
    assert response.status_code == 422
    source["connector"].assert_not_called()
    source["predictor"].assert_not_called()

def test_predict_endpoint_rejects_missing_required_field(prediction_api, prediction_body):
    source = prediction_api
    del prediction_body["player_id"]
    response = source["client"].post("/predict", json=prediction_body)
    assert response.status_code == 422
    source["connector"].assert_not_called()
    source["predictor"].assert_not_called()

@pytest.mark.parametrize("missing_resource", ["database", "model"])
def test_predict_endpoint_returns_503_for_missing_resources(prediction_api, prediction_body, missing_resource):
    source = prediction_api
    if missing_resource == "database":
        source["db_path"].unlink()
    else:
        source["configs"]["linear_regression"][0].unlink()
    response = source["client"].post("/predict", json=prediction_body)
    assert response.status_code == 503
    assert response.json() == {"detail": "Prediction service data or model is unavailable"}
    source["connector"].assert_not_called()
    source["predictor"].assert_not_called()

def test_predict_endpoint_returns_404_for_unknown_player(prediction_api, prediction_body):
    source = prediction_api
    source["connection"].execute.return_value.fetchone.return_value = None
    response = source["client"].post("/predict", json=prediction_body)
    assert response.status_code == 404
    assert response.json() == {"detail": "Player not found"}
    source["predictor"].assert_not_called()
    source["connection"].close.assert_called_once_with()

def test_predict_endpoint_returns_422_for_unavailable_history(prediction_api, prediction_body):
    source = prediction_api
    message = "No completed games before prediction date"
    source["predictor"].side_effect = ValueError(message)
    response = source["client"].post("/predict", json=prediction_body)
    assert response.status_code == 422
    assert response.json() == {"detail": message}
    source["connection"].close.assert_called_once_with()

def test_predict_endpoint_closes_connection_after_unexpected_failure(prediction_api, prediction_body):
    source = prediction_api
    source["predictor"].side_effect = RuntimeError("Unexpected inference failure")
    with pytest.raises(RuntimeError, match="Unexpected inference failure"):
        source["client"].post("/predict", json=prediction_body)
    source["connection"].close.assert_called_once_with()