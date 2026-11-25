from unittest.mock import Mock, call
import psycopg
import pytest
from fastapi.testclient import TestClient
from fantasy_basketball import api as api_module

@pytest.fixture(params=["/predict", "/lineups/daily", "/lineups/weekly", "/matchups"])
def postgres_api_case(request, monkeypatch, tmp_path):
    game_date = "2026-02-03"
    player_a = {"player_id": 10, "name": "Player A", "positions": ["PG"], "team": "DEN", "schedule": [game_date]}
    player_b = {"player_id": 20, "name": "Player B", "positions": ["C"], "team": "GSW", "schedule": [game_date]}
    common = {"season": "2025-26"}
    bodies = {
        "/predict": {**common, "player_id": 10, "game_date": game_date},
        "/lineups/daily": {**common, "players": [player_a], "game_date": game_date, "lineup_slots": ["PG"]},
        "/lineups/weekly": {**common, "players": [player_a], "start_date": game_date, "end_date": game_date, "lineup_slots": ["PG"]},
        "/matchups": {**common, "team_a": [player_a], "team_b": [player_b], "start_date": game_date, "end_date": game_date, "lineup_slots": ["PG"]}}
    weekly_results = {game_date: {"lineup": {}, "projected_points": 0.0}}
    predictors = {
        "/predict": Mock(return_value=42.5),
        "/lineups/daily": Mock(return_value=({}, 0.0)),
        "/lineups/weekly": Mock(return_value=(weekly_results, 0.0)),
        "/matchups": Mock(return_value={
            "team_a": {"weekly_results": weekly_results, "projected_total": 0.0},
            "team_b": {"weekly_results": weekly_results, "projected_total": 0.0},
            "projected_winner": "Tie",
            "point_diff": 0.0})}
    connection = Mock(spec=psycopg.Connection)
    connection.execute.return_value.fetchone.return_value = (10,)
    postgres_connector = Mock(return_value=connection)
    sqlite_connector = Mock()
    database_url = "postgresql://user:password@localhost/fantasy_test"
    model_configs = {}
    for name, filename, model_type in [
        ("linear_regression", "linear.pkl", "sklearn"),
        ("random_forest", "forest.pkl", "sklearn"),
        ("pytorch", "network.pt", "pytorch")]:
        path = tmp_path / filename
        path.touch()
        model_configs[name] = (path, model_type)
    monkeypatch.setattr(api_module, "DATABASE_URL", database_url)
    monkeypatch.setattr(api_module, "DB_PATH", tmp_path / "missing.db")
    monkeypatch.setattr(api_module, "MODEL_CONFIGS", model_configs)
    monkeypatch.setattr(api_module, "get_postgres_connection", postgres_connector)
    monkeypatch.setattr(api_module, "get_connection", sqlite_connector)
    monkeypatch.setattr(api_module, "predict_stored_player_next_game", predictors["/predict"])
    monkeypatch.setattr(api_module, "optimize_projected_day", predictors["/lineups/daily"])
    monkeypatch.setattr(api_module, "optimize_projected_week", predictors["/lineups/weekly"])
    monkeypatch.setattr(api_module, "analyze_projected_matchup", predictors["/matchups"])
    with TestClient(api_module.app) as client:
        yield {
            "client": client,
            "endpoint": request.param,
            "body": bodies[request.param],
            "connection": connection,
            "postgres_connector": postgres_connector,
            "sqlite_connector": sqlite_connector,
            "database_url": database_url,
            "model_configs": model_configs,
            "predictor": predictors[request.param]}

@pytest.mark.parametrize("model_name", ["linear_regression", "random_forest", "pytorch"])
def test_endpoints_use_postgres_without_sqlite_file(postgres_api_case, model_name):
    case = postgres_api_case
    body = {**case["body"], "model": model_name}
    response = case["client"].post(case["endpoint"], json=body)
    assert response.status_code == 200, response.text
    assert response.json()["model"] == model_name
    assert response.json()["as_of_date"] == "2026-02-03"
    case["postgres_connector"].assert_called_once_with(case["database_url"])
    case["sqlite_connector"].assert_not_called()
    expected_queries = [call("SELECT player_id FROM players WHERE player_id = %s", (10,))]
    if case["endpoint"] == "/matchups":
        expected_queries.append(call("SELECT player_id FROM players WHERE player_id = %s", (20,)))
    assert case["connection"].execute.call_args_list == expected_queries
    case["predictor"].assert_called_once()
    args = case["predictor"].call_args.args
    model_path, model_type = case["model_configs"][model_name]
    assert args[0] is case["connection"]
    assert args[-2] == str(model_path)
    assert args[-1] == model_type
    assert case["predictor"].call_args.kwargs["as_of_date"] == "2026-02-03"
    case["connection"].close.assert_called_once_with()

def test_postgres_endpoints_return_404_for_missing_player(postgres_api_case):
    case = postgres_api_case
    case["connection"].execute.return_value.fetchone.return_value = None
    response = case["client"].post(case["endpoint"], json=case["body"])
    assert response.status_code == 404
    assert "Player not found" in response.json()["detail"]
    case["predictor"].assert_not_called()
    case["sqlite_connector"].assert_not_called()
    case["connection"].close.assert_called_once_with()

def test_postgres_endpoints_return_503_for_connection_failure(postgres_api_case):
    case = postgres_api_case
    case["postgres_connector"].side_effect = psycopg.OperationalError("Private connection details")
    response = case["client"].post(case["endpoint"], json=case["body"])
    assert response.status_code == 503
    assert response.json()["detail"] == "Database service is unavailable"
    case["predictor"].assert_not_called()
    case["sqlite_connector"].assert_not_called()