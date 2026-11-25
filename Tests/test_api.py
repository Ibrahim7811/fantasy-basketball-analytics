from fastapi.testclient import TestClient
from fantasy_basketball.api import app
from unittest.mock import Mock
import pytest
from fantasy_basketball import api as api_module
from fantasy_basketball.player_class import Player
from datetime import date
import psycopg

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

@pytest.fixture
def daily_lineup_body():
    return {
        "players": [
            {
                "player_id": 201939,
                "name": "Stephen Curry",
                "positions": ["PG"],
                "team": "GSW",
                "schedule": ["2026-02-05", "2026-02-03"]},
            {
                "player_id": 203999,
                "name": "Nikola Jokic",
                "positions": ["C"],
                "team": "DEN",
                "schedule": ["2026-02-05"]}],
        "season": "2025-26",
        "game_date": "2026-02-05",
        "lineup_slots": ["PG", "C1", "SG"]}

@pytest.fixture
def daily_lineup_api(prediction_api, monkeypatch):
    source = prediction_api
    lineup = {
        "PG": Player("Stephen Curry", ["PG"], 42.5, ["2026-02-05"], "GSW"),
        "C1": Player("Nikola Jokic", ["C"], 57.5, ["2026-02-05"], "DEN")}
    optimizer = Mock(return_value=(lineup, 100.0))
    monkeypatch.setattr(api_module, "optimize_projected_day", optimizer)
    return {**source, "optimizer": optimizer}

@pytest.mark.parametrize("model", ["linear_regression", "random_forest", "pytorch"])
@pytest.mark.parametrize("cutoff", [None, "2026-02-01"])
def test_daily_lineup_endpoint_converts_roster_and_returns_lineup(daily_lineup_api, daily_lineup_body, model, cutoff):
    source = daily_lineup_api
    payload = {**daily_lineup_body, "model": model}
    if cutoff is not None:
        payload["as_of_date"] = cutoff
    response = source["client"].post("/lineups/daily", json=payload)
    effective_cutoff = cutoff or "2026-02-05"
    assert response.status_code == 200
    assert response.json() == {
        "season": "2025-26",
        "game_date": "2026-02-05",
        "model": model,
        "as_of_date": effective_cutoff,
        "lineup": {
            "PG": {
                "player_id": 201939,
                "name": "Stephen Curry",
                "positions": ["PG"],
                "team": "GSW",
                "projected_fp": 42.5},
            "C1": {
                "player_id": 203999,
                "name": "Nikola Jokic",
                "positions": ["C"],
                "team": "DEN",
                "projected_fp": 57.5},
            "SG": None},
        "projected_total": 100.0}
    source["connector"].assert_called_once_with(str(source["db_path"]))
    source["optimizer"].assert_called_once()
    args = source["optimizer"].call_args.args
    kwargs = source["optimizer"].call_args.kwargs
    assert args[0] is source["connection"]
    players = args[1]
    assert len(players) == 2
    assert all(isinstance(player, Player) for player in players)
    assert [player._name for player in players] == ["Stephen Curry", "Nikola Jokic"]
    assert [player._positions for player in players] == [["PG"], ["C"]]
    assert [player._team for player in players] == ["GSW", "DEN"]
    assert all(player._projected_points == 0.0 for player in players)
    assert players[0]._schedule == ["2026-02-03", "2026-02-05"]
    assert players[1]._schedule == ["2026-02-05"]
    model_path, model_type = source["configs"][model]
    assert args[2:] == (
        {"Stephen Curry": 201939,
         "Nikola Jokic": 203999},
        "2025-26",
        "2026-02-05",
        ["PG", "C1", "SG"],
        str(model_path),
        model_type)
    assert kwargs == {"as_of_date": effective_cutoff}
    assert [item.args[1] for item in source["connection"].execute.call_args_list] == [(201939,), (203999,)]
    source["connection"].close.assert_called_once_with()

def test_daily_lineup_endpoint_defaults_to_linear_regression(daily_lineup_api, daily_lineup_body):
    source = daily_lineup_api
    response = source["client"].post("/lineups/daily", json=daily_lineup_body)
    assert response.status_code == 200
    assert response.json()["model"] == "linear_regression"
    assert response.json()["as_of_date"] == "2026-02-05"

def test_daily_lineup_endpoint_returns_null_for_every_unfilled_slot(daily_lineup_api, daily_lineup_body):
    source = daily_lineup_api
    source["optimizer"].return_value = ({}, 0.0)
    response = source["client"].post("/lineups/daily", json=daily_lineup_body)
    assert response.status_code == 200
    assert response.json()["lineup"] == {"PG": None, "C1": None, "SG": None}
    assert response.json()["projected_total"] == 0.0
    source["connection"].close.assert_called_once_with()

@pytest.mark.parametrize("problem", ["duplicate_id", "duplicate_name", "future_cutoff", "invalid_position"])
def test_daily_lineup_endpoint_rejects_invalid_roster_before_database_access(daily_lineup_api, daily_lineup_body, problem):
    source = daily_lineup_api
    if problem == "duplicate_id":
        daily_lineup_body["players"][1]["player_id"] = 201939
    elif problem == "duplicate_name":
        daily_lineup_body["players"][1]["name"] = "  STEPHEN CURRY  "
    elif problem == "future_cutoff":
        daily_lineup_body["as_of_date"] = "2026-02-06"
    else:
        daily_lineup_body["players"][0]["positions"] = ["INVALID"]
    response = source["client"].post("/lineups/daily", json=daily_lineup_body)
    assert response.status_code == 422
    source["connector"].assert_not_called()
    source["optimizer"].assert_not_called()

@pytest.mark.parametrize("missing_resource", ["database", "model"])
def test_daily_lineup_endpoint_returns_503_for_missing_resources(daily_lineup_api, daily_lineup_body, missing_resource):
    source = daily_lineup_api
    if missing_resource == "database":
        source["db_path"].unlink()
    else:
        source["configs"]["linear_regression"][0].unlink()
    response = source["client"].post("/lineups/daily", json=daily_lineup_body)
    assert response.status_code == 503
    assert response.json() == {"detail": "Prediction service data or model is unavailable"}
    source["connector"].assert_not_called()
    source["optimizer"].assert_not_called()

def test_daily_lineup_endpoint_returns_404_for_unknown_roster_player(daily_lineup_api, daily_lineup_body):
    source = daily_lineup_api
    source["connection"].execute.return_value.fetchone.side_effect = [(201939,), None]
    response = source["client"].post("/lineups/daily", json=daily_lineup_body)
    assert response.status_code == 404
    assert response.json() == {"detail": "Player not found: 203999"}
    source["optimizer"].assert_not_called()
    source["connection"].close.assert_called_once_with()

def test_daily_lineup_endpoint_returns_422_for_inference_error(daily_lineup_api, daily_lineup_body):
    source = daily_lineup_api
    message = "No completed games before prediction date"
    source["optimizer"].side_effect = ValueError(message)
    response = source["client"].post("/lineups/daily", json=daily_lineup_body)
    assert response.status_code == 422
    assert response.json() == {"detail": message}
    source["connection"].close.assert_called_once_with()

def test_daily_lineup_endpoint_closes_connection_after_unexpected_failure(daily_lineup_api, daily_lineup_body):
    source = daily_lineup_api
    source["optimizer"].side_effect = RuntimeError("Unexpected optimization failure")
    with pytest.raises(RuntimeError, match="Unexpected optimization failure"):
        source["client"].post("/lineups/daily", json=daily_lineup_body)
    source["connection"].close.assert_called_once_with()

@pytest.fixture
def weekly_lineup_body(daily_lineup_body):
    return {
        "players": daily_lineup_body["players"],
        "season": "2025-26",
        "start_date": "2026-02-03",
        "end_date": "2026-02-05",
        "lineup_slots": ["PG", "C1", "SG"]}

@pytest.fixture
def weekly_lineup_api(prediction_api, monkeypatch):
    curry_first = Player("Stephen Curry", ["PG"], 40.0, ["2026-02-03"], "GSW")
    curry_last = Player("Stephen Curry", ["PG"], 42.5, ["2026-02-05"], "GSW")
    jokic = Player("Nikola Jokic", ["C"], 57.5, ["2026-02-05"], "DEN")
    # Deliberately unsorted to check chronological response formatting.
    results = {
        "2026-02-05": {"lineup": {"PG": curry_last, "C1": jokic}, "projected_points": 100.0},
        "2026-02-04": {"lineup": {}, "projected_points": 0.0},
        "2026-02-03": {"lineup": {"PG": curry_first}, "projected_points": 40.0}}
    optimizer = Mock(return_value=(results, 140.0))
    monkeypatch.setattr(api_module, "optimize_projected_week", optimizer)
    return {**prediction_api, "optimizer": optimizer}

@pytest.mark.parametrize("model", ["linear_regression", "random_forest", "pytorch"])
@pytest.mark.parametrize("cutoff", [None, "2026-02-01"])
def test_weekly_lineup_endpoint_returns_sorted_days_and_fixed_cutoff(weekly_lineup_api, weekly_lineup_body, model, cutoff):
    source = weekly_lineup_api
    payload = {**weekly_lineup_body, "model": model}
    if cutoff is not None:
        payload["as_of_date"] = cutoff
    response = source["client"].post("/lineups/weekly", json=payload)
    effective_cutoff = cutoff or "2026-02-03"
    assert response.status_code == 200
    result = response.json()
    assert result["season"] == "2025-26"
    assert result["start_date"] == "2026-02-03"
    assert result["end_date"] == "2026-02-05"
    assert result["model"] == model
    assert result["as_of_date"] == effective_cutoff
    assert result["projected_total"] == pytest.approx(140.0)
    days = result["daily_results"]
    assert [day["game_date"] for day in days] == ["2026-02-03", "2026-02-04", "2026-02-05"]
    assert [day["projected_total"] for day in days] == [40.0, 0.0, 100.0]
    for day in days:
        assert day["season"] == "2025-26"
        assert day["model"] == model
        assert day["as_of_date"] == effective_cutoff
        assert set(day["lineup"]) == {"PG", "C1", "SG"}
    assert days[0]["lineup"]["PG"] == {
        "player_id": 201939,
        "name": "Stephen Curry",
        "positions": ["PG"],
        "team": "GSW",
        "projected_fp": 40.0}
    assert days[0]["lineup"]["C1"] is None
    assert days[1]["lineup"] == {"PG": None, "C1": None, "SG": None}
    assert days[2]["lineup"]["PG"]["projected_fp"] == 42.5
    assert days[2]["lineup"]["C1"]["player_id"] == 203999
    assert days[2]["lineup"]["C1"]["projected_fp"] == 57.5
    assert days[2]["lineup"]["SG"] is None
    assert sum(day["projected_total"] for day in days) == pytest.approx(result["projected_total"])
    source["connector"].assert_called_once_with(str(source["db_path"]))
    source["optimizer"].assert_called_once()
    args = source["optimizer"].call_args.args
    assert args[0] is source["connection"]
    players = args[1]
    assert [player._name for player in players] == ["Stephen Curry", "Nikola Jokic"]
    assert [player._positions for player in players] == [["PG"], ["C"]]
    assert [player._team for player in players] == ["GSW", "DEN"]
    assert all(player._projected_points == 0.0 for player in players)
    assert players[0]._schedule == ["2026-02-03", "2026-02-05"]
    assert players[1]._schedule == ["2026-02-05"]
    assert args[2] == {"Stephen Curry": 201939, "Nikola Jokic": 203999}
    model_path, model_type = source["configs"][model]
    assert args[3:] == (
        "2025-26",
        "2026-02-03",
        "2026-02-05",
        ["PG", "C1", "SG"],
        str(model_path),
        model_type)
    assert source["optimizer"].call_args.kwargs == {"as_of_date": effective_cutoff}
    assert [item.args[1] for item in source["connection"].execute.call_args_list] == [(201939,), (203999,)]
    source["connection"].close.assert_called_once_with()

def test_weekly_lineup_endpoint_defaults_to_linear_regression(weekly_lineup_api, weekly_lineup_body):
    source = weekly_lineup_api
    response = source["client"].post("/lineups/weekly", json=weekly_lineup_body)
    assert response.status_code == 200
    assert response.json()["model"] == "linear_regression"
    assert response.json()["as_of_date"] == "2026-02-03"

@pytest.mark.parametrize("field, value", [
    ("start_date", "2026-02-06"),
    ("end_date", "not-a-date"),
    ("as_of_date", "2026-02-04"),
    ("players", []),
    ("lineup_slots", ["PG", "PG"]),
    ("model", "unknown")
])
def test_weekly_lineup_endpoint_rejects_invalid_requests_before_database_access(weekly_lineup_api, weekly_lineup_body, field, value):
    source = weekly_lineup_api
    response = source["client"].post("/lineups/weekly", json={**weekly_lineup_body, field: value})
    assert response.status_code == 422
    source["connector"].assert_not_called()
    source["optimizer"].assert_not_called()

@pytest.mark.parametrize("missing_resource", ["database", "model"])
def test_weekly_lineup_endpoint_returns_503_for_missing_resources(weekly_lineup_api, weekly_lineup_body, missing_resource):
    source = weekly_lineup_api
    if missing_resource == "database":
        source["db_path"].unlink()
    else:
        source["configs"]["linear_regression"][0].unlink()
    response = source["client"].post("/lineups/weekly", json=weekly_lineup_body)
    assert response.status_code == 503
    assert response.json() == {"detail": "Prediction service data or model is unavailable"}
    source["connector"].assert_not_called()
    source["optimizer"].assert_not_called()

def test_weekly_lineup_endpoint_returns_404_for_unknown_player(weekly_lineup_api, weekly_lineup_body):
    source = weekly_lineup_api
    source["connection"].execute.return_value.fetchone.side_effect = [(201939,), None]
    response = source["client"].post("/lineups/weekly", json=weekly_lineup_body)
    assert response.status_code == 404
    assert response.json() == {"detail": "Player not found: 203999"}
    source["optimizer"].assert_not_called()
    source["connection"].close.assert_called_once_with()

def test_weekly_lineup_endpoint_returns_422_for_inference_error(weekly_lineup_api, weekly_lineup_body):
    source = weekly_lineup_api
    message = "No completed games before prediction date"
    source["optimizer"].side_effect = ValueError(message)
    response = source["client"].post("/lineups/weekly", json=weekly_lineup_body)
    assert response.status_code == 422
    assert response.json() == {"detail": message}
    source["connection"].close.assert_called_once_with()

def test_weekly_lineup_endpoint_closes_connection_after_unexpected_failure(weekly_lineup_api, weekly_lineup_body):
    source = weekly_lineup_api
    source["optimizer"].side_effect = RuntimeError("Unexpected optimization failure")
    with pytest.raises(RuntimeError, match="Unexpected optimization failure"):
        source["client"].post("/lineups/weekly", json=weekly_lineup_body)
    source["connection"].close.assert_called_once_with()

@pytest.fixture
def matchup_body(weekly_lineup_body):
    return {
        "team_a": [weekly_lineup_body["players"][0]],
        "team_b": [weekly_lineup_body["players"][1]],
        "season": "2025-26",
        "start_date": "2026-02-03",
        "end_date": "2026-02-05",
        "lineup_slots": ["PG", "C1", "SG"]
    }

@pytest.fixture
def matchup_api(prediction_api, monkeypatch):
    curry_first = Player("Stephen Curry", ["PG"], 40.0, ["2026-02-03"], "GSW")
    curry_last = Player("Stephen Curry", ["PG"], 42.5, ["2026-02-05"], "GSW")
    jokic = Player("Nikola Jokic", ["C"], 57.5, ["2026-02-05"], "DEN")
    result = {
        "team_a": {
            "weekly_results": {
                "2026-02-05": {"lineup": {"PG": curry_last}, "projected_points": 42.5},
                "2026-02-04": {"lineup": {}, "projected_points": 0.0},
                "2026-02-03": {"lineup": {"PG": curry_first}, "projected_points": 40.0}},
            "projected_total": 82.5},
        "team_b": {
            "weekly_results": {
                "2026-02-05": {"lineup": {"C1": jokic}, "projected_points": 57.5},
                "2026-02-04": {"lineup": {}, "projected_points": 0.0},
                "2026-02-03": {"lineup": {}, "projected_points": 0.0}},
            "projected_total": 57.5},
        "projected_winner": "Team A",
        "point_diff": 25.0}
    analyzer = Mock(return_value=result)
    monkeypatch.setattr(api_module, "analyze_projected_matchup", analyzer)
    return {**prediction_api, "analyzer": analyzer}

@pytest.mark.parametrize("model", ["linear_regression", "random_forest", "pytorch"])
@pytest.mark.parametrize("cutoff", [None, "2026-02-01"])
def test_matchup_endpoint_returns_teams_and_forwards_shared_configuration(matchup_api, matchup_body, model, cutoff):
    source = matchup_api
    payload = {**matchup_body, "model": model}
    if cutoff is not None:
        payload["as_of_date"] = cutoff
    response = source["client"].post("/matchups", json=payload)
    effective_cutoff = cutoff or "2026-02-03"
    assert response.status_code == 200
    result = response.json()
    assert result["season"] == "2025-26"
    assert result["start_date"] == "2026-02-03"
    assert result["end_date"] == "2026-02-05"
    assert result["model"] == model
    assert result["as_of_date"] == effective_cutoff
    assert result["projected_winner"] == "Team A"
    assert result["point_diff"] == pytest.approx(25.0)
    assert result["team_a"]["projected_total"] == pytest.approx(82.5)
    assert result["team_b"]["projected_total"] == pytest.approx(57.5)
    for team_key in ("team_a", "team_b"):
        team = result[team_key]
        assert team["season"] == "2025-26"
        assert team["start_date"] == "2026-02-03"
        assert team["end_date"] == "2026-02-05"
        assert team["model"] == model
        assert team["as_of_date"] == effective_cutoff
        assert [day["game_date"] for day in team["daily_results"]] == ["2026-02-03", "2026-02-04", "2026-02-05"]
        assert sum(day["projected_total"] for day in team["daily_results"]) == pytest.approx(team["projected_total"])
        for day in team["daily_results"]:
            assert day["model"] == model
            assert day["as_of_date"] == effective_cutoff
            assert set(day["lineup"]) == {"PG", "C1", "SG"}
        assert team["daily_results"][1]["lineup"] == {"PG": None, "C1": None, "SG": None}
    selected_a = result["team_a"]["daily_results"][0]["lineup"]["PG"]
    selected_b = result["team_b"]["daily_results"][2]["lineup"]["C1"]
    assert selected_a["player_id"] == 201939
    assert selected_a["name"] == "Stephen Curry"
    assert selected_a["projected_fp"] == 40.0
    assert selected_b["player_id"] == 203999
    assert selected_b["name"] == "Nikola Jokic"
    assert selected_b["projected_fp"] == 57.5
    source["analyzer"].assert_called_once()
    args = source["analyzer"].call_args.args
    assert args[0] is source["connection"]
    team_a, team_b = args[1:3]
    assert len(team_a) == len(team_b) == 1
    assert team_a[0]._name == "Stephen Curry"
    assert team_b[0]._name == "Nikola Jokic"
    assert team_a[0]._positions == ["PG"]
    assert team_b[0]._positions == ["C"]
    assert team_a[0]._schedule == ["2026-02-03", "2026-02-05"]
    assert team_b[0]._schedule == ["2026-02-05"]
    assert team_a[0]._projected_points == team_b[0]._projected_points == 0.0
    assert args[3] == {"Stephen Curry": 201939, "Nikola Jokic": 203999}
    model_path, model_type = source["configs"][model]
    assert args[4:] == (
        "2025-26",
        "2026-02-03",
        "2026-02-05",
        ["PG", "C1", "SG"],
        str(model_path),
        model_type)
    assert source["analyzer"].call_args.kwargs == {"as_of_date": effective_cutoff}
    assert [item.args[1] for item in source["connection"].execute.call_args_list] == [(201939,), (203999,)]
    source["connector"].assert_called_once_with(str(source["db_path"]))
    source["connection"].close.assert_called_once_with()

def test_matchup_endpoint_defaults_to_linear_regression(matchup_api, matchup_body):
    source = matchup_api
    response = source["client"].post("/matchups", json=matchup_body)
    assert response.status_code == 200
    assert response.json()["model"] == "linear_regression"
    assert response.json()["as_of_date"] == "2026-02-03"

@pytest.mark.parametrize("points_a, points_b, winner", [
    (40.0, 60.0, "Team B"),
    (50.0, 50.0, "Tie")
])
def test_matchup_endpoint_returns_other_winner_outcomes(matchup_api, matchup_body, points_a, points_b, winner):
    source = matchup_api
    result = source["analyzer"].return_value
    for team_key, points in (("team_a", points_a), ("team_b", points_b)):
        team = result[team_key]
        team["projected_total"] = points
        active_days = [day for day in team["weekly_results"].values() if day["lineup"]]
        for day in active_days:
            day["projected_points"] = points / len(active_days)
            for player in day["lineup"].values():
                player._projected_points = day["projected_points"]
    result["projected_winner"] = winner
    result["point_diff"] = points_a - points_b
    response = source["client"].post("/matchups", json=matchup_body)
    assert response.status_code == 200
    assert response.json()["projected_winner"] == winner
    assert response.json()["point_diff"] == pytest.approx(points_a - points_b)

@pytest.mark.parametrize("problem", ["overlapping_player", "empty_team", "late_cutoff", "duplicate_slots"])
def test_matchup_endpoint_rejects_invalid_requests_before_database_access(matchup_api, matchup_body, problem):
    source = matchup_api
    if problem == "overlapping_player":
        matchup_body["team_b"][0]["player_id"] = matchup_body["team_a"][0]["player_id"]
    elif problem == "empty_team":
        matchup_body["team_b"] = []
    elif problem == "late_cutoff":
        matchup_body["as_of_date"] = "2026-02-04"
    else:
        matchup_body["lineup_slots"] = ["PG", "PG"]
    response = source["client"].post("/matchups", json=matchup_body)
    assert response.status_code == 422
    source["connector"].assert_not_called()
    source["analyzer"].assert_not_called()

@pytest.mark.parametrize("missing_resource", ["database", "model"])
def test_matchup_endpoint_returns_503_for_missing_resources(matchup_api, matchup_body, missing_resource):
    source = matchup_api
    if missing_resource == "database":
        source["db_path"].unlink()
    else:
        source["configs"]["linear_regression"][0].unlink()
    response = source["client"].post("/matchups", json=matchup_body)
    assert response.status_code == 503
    assert response.json() == {"detail": "Prediction service data or model is unavailable"}
    source["connector"].assert_not_called()
    source["analyzer"].assert_not_called()

@pytest.mark.parametrize("missing_team", ["team_a", "team_b"])
def test_matchup_endpoint_returns_404_for_unknown_player_on_either_team(matchup_api, matchup_body, missing_team):
    source = matchup_api
    if missing_team == "team_a":
        source["connection"].execute.return_value.fetchone.side_effect = [None]
        missing_id = 201939
    else:
        source["connection"].execute.return_value.fetchone.side_effect = [(201939,), None]
        missing_id = 203999
    response = source["client"].post("/matchups", json=matchup_body)
    assert response.status_code == 404
    assert response.json() == {"detail": f"Player not found: {missing_id}"}
    source["analyzer"].assert_not_called()
    source["connection"].close.assert_called_once_with()

def test_matchup_endpoint_returns_422_for_inference_error(matchup_api, matchup_body):
    source = matchup_api
    message = "No completed games before prediction date"
    source["analyzer"].side_effect = ValueError(message)
    response = source["client"].post("/matchups", json=matchup_body)
    assert response.status_code == 422
    assert response.json() == {"detail": message}
    source["connection"].close.assert_called_once_with()

def test_matchup_endpoint_closes_connection_after_unexpected_failure(matchup_api, matchup_body):
    source = matchup_api
    source["analyzer"].side_effect = RuntimeError("Unexpected matchup failure")
    with pytest.raises(RuntimeError, match="Unexpected matchup failure"):
        source["client"].post("/matchups", json=matchup_body)
    source["connection"].close.assert_called_once_with()

def test_get_api_connection_uses_postgres_when_configured(monkeypatch):
    database_url = "postgresql://user:password@localhost/fantasy_test"
    connection = Mock()
    postgres_connector = Mock(return_value=connection)
    sqlite_connector = Mock()
    monkeypatch.setattr(api_module, "DATABASE_URL", database_url)
    monkeypatch.setattr(api_module, "get_postgres_connection", postgres_connector)
    monkeypatch.setattr(api_module, "get_connection", sqlite_connector)
    result = api_module.get_api_connection()
    assert result is connection
    postgres_connector.assert_called_once_with(database_url)
    sqlite_connector.assert_not_called()

def test_get_api_connection_uses_sqlite_without_database_url(monkeypatch, tmp_path):
    database_path = tmp_path / "fantasy.db"
    connection = Mock()
    sqlite_connector = Mock(return_value=connection)
    postgres_connector = Mock()
    monkeypatch.setattr(api_module, "DATABASE_URL", "")
    monkeypatch.setattr(api_module, "DB_PATH", database_path)
    monkeypatch.setattr(api_module, "get_connection", sqlite_connector)
    monkeypatch.setattr(api_module, "get_postgres_connection", postgres_connector)
    result = api_module.get_api_connection()
    assert result is connection
    sqlite_connector.assert_called_once_with(str(database_path))
    postgres_connector.assert_not_called()

def test_get_api_connection_returns_503_when_postgres_is_unavailable(monkeypatch):
    postgres_connector = Mock(side_effect=psycopg.OperationalError("Private connection details"))
    sqlite_connector = Mock()
    monkeypatch.setattr(api_module, "DATABASE_URL", "postgresql://localhost/fantasy_test")
    monkeypatch.setattr(api_module, "get_postgres_connection", postgres_connector)
    monkeypatch.setattr(api_module, "get_connection", sqlite_connector)
    with pytest.raises(api_module.HTTPException) as caught:
        api_module.get_api_connection()
    assert caught.value.status_code == 503
    assert caught.value.detail == "Database service is unavailable"
    sqlite_connector.assert_not_called()