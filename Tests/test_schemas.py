from datetime import date
import pytest
from pydantic import ValidationError
from copy import deepcopy
from fantasy_basketball.schemas import (
    PredictionRequest,
    PredictionResponse,
    RosterPlayerRequest,
    DailyLineupRequest,
    ProjectedPlayerResponse,
    DailyLineupResponse,
    WeeklyLineupRequest,
    WeeklyLineupResponse)

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
    assert any(item["loc"] == (field,) for item in error.value.errors())\

@pytest.fixture
def roster_player_payload():
    return {
        "player_id": 203999,
        "name": "Nikola Jokic",
        "positions": ["C"],
        "team": "DEN",
        "schedule": ["2026-02-05", "2026-02-03"]}

def test_roster_player_parses_dates_and_sorts_schedule(roster_player_payload):
    player = RosterPlayerRequest(**roster_player_payload)
    assert player.player_id == 203999
    assert player.name == "Nikola Jokic"
    assert player.positions == ["C"]
    assert player.team == "DEN"
    assert player.schedule == [date(2026, 2, 3), date(2026, 2, 5)]
    assert player.model_dump(mode="json") == {**roster_player_payload,"schedule": ["2026-02-03", "2026-02-05"]}

def test_roster_player_trims_name(roster_player_payload):
    player = RosterPlayerRequest(**{**roster_player_payload, "name": "  Nikola Jokic  "})
    assert player.name == "Nikola Jokic"

@pytest.mark.parametrize("positions", [["PG"], ["SG", "SF"], ["PF", "C"]])
def test_roster_player_accepts_valid_positions(roster_player_payload, positions):
    player = RosterPlayerRequest(**{**roster_player_payload, "positions": positions})
    assert player.positions == positions

def test_roster_player_default_schedules_are_independent(roster_player_payload):
    del roster_player_payload["schedule"]
    first = RosterPlayerRequest(**roster_player_payload)
    second = RosterPlayerRequest(**roster_player_payload)
    assert first.schedule == []
    assert second.schedule == []
    first.schedule.append(date(2026, 2, 3))
    assert second.schedule == []

@pytest.mark.parametrize("field", ["player_id", "name", "positions", "team"])
def test_roster_player_requires_identity_and_eligibility(roster_player_payload, field):
    del roster_player_payload[field]
    with pytest.raises(ValidationError) as error:
        RosterPlayerRequest(**roster_player_payload)
    assert any(item["loc"] == (field,) and item["type"] == "missing" for item in error.value.errors())

@pytest.mark.parametrize("field, value", [
    ("player_id", 0),
    ("player_id", -1),
    ("name", ""),
    ("name", "   "),
    ("positions", []),
    ("positions", ["PG", "PG"]),
    ("positions", ["UTIL"]),
    ("positions", ["pg"]),
    ("team", "XYZ"),
    ("team", "den"),
    ("team", "DENV"),
    ("schedule", ["2026-02-03", "2026-02-03"]),
    ("schedule", ["2026-02-30"]),
    ("schedule", ["not-a-date"])
])
def test_roster_player_rejects_invalid_fields(roster_player_payload, field, value):
    with pytest.raises(ValidationError) as error:
        RosterPlayerRequest(**{**roster_player_payload, field: value})
    assert any(item["loc"][0] == field for item in error.value.errors())

def test_roster_player_preserves_input_payload(roster_player_payload):
    original_positions = roster_player_payload["positions"].copy()
    original_schedule = roster_player_payload["schedule"].copy()
    player = RosterPlayerRequest(**roster_player_payload)
    assert roster_player_payload["positions"] == original_positions
    assert roster_player_payload["schedule"] == original_schedule
    player.positions.append("PF")
    assert roster_player_payload["positions"] == original_positions

@pytest.fixture
def daily_lineup_payload(roster_player_payload):
    return {
        "players": [deepcopy(roster_player_payload)],
        "season": "2025-26",
        "game_date": "2026-02-05",
        "lineup_slots": ["C1", "UTIL1"]}

def test_daily_lineup_request_parses_roster_and_sets_defaults(daily_lineup_payload):
    request = DailyLineupRequest(**daily_lineup_payload)
    assert isinstance(request.players[0], RosterPlayerRequest)
    assert request.players[0].name == "Nikola Jokic"
    assert request.players[0].schedule == [date(2026, 2, 3), date(2026, 2, 5)]
    assert request.game_date == date(2026, 2, 5)
    assert request.lineup_slots == ["C1", "UTIL1"]
    assert request.model == "linear_regression"
    assert request.as_of_date is None

@pytest.mark.parametrize("model", ["linear_regression", "random_forest", "pytorch"])
def test_daily_lineup_request_accepts_supported_models(daily_lineup_payload, model):
    request = DailyLineupRequest(**{**daily_lineup_payload, "model": model})
    assert request.model == model

@pytest.mark.parametrize("cutoff", ["2026-02-01", "2026-02-05"])
def test_daily_lineup_request_accepts_valid_cutoffs(daily_lineup_payload, cutoff):
    request = DailyLineupRequest(**{**daily_lineup_payload, "as_of_date": cutoff})
    assert request.as_of_date == date.fromisoformat(cutoff)

def test_daily_lineup_request_accepts_all_supported_slots(daily_lineup_payload):
    slots = ["PG", "SG", "G", "SF", "PF", "F", "C1", "C2", "UTIL1", "UTIL2"]
    request = DailyLineupRequest(**{**daily_lineup_payload, "lineup_slots": slots})
    assert request.lineup_slots == slots

@pytest.mark.parametrize("field", ["players", "season", "game_date", "lineup_slots"])
def test_daily_lineup_request_requires_fields(daily_lineup_payload, field):
    del daily_lineup_payload[field]
    with pytest.raises(ValidationError) as error:
        DailyLineupRequest(**daily_lineup_payload)
    assert any(item["loc"] == (field,) and item["type"] == "missing" for item in error.value.errors())

@pytest.mark.parametrize("field, value", [
    ("players", []),
    ("season", "2025"),
    ("game_date", "2026-02-30"),
    ("lineup_slots", []),
    ("lineup_slots", ["PG", "PG"]),
    ("lineup_slots", ["C"]),
    ("lineup_slots", ["pg"]),
    ("lineup_slots", ["UNKNOWN"]),
    ("model", "unknown"),
    ("as_of_date", "not-a-date")
])
def test_daily_lineup_request_rejects_invalid_fields(daily_lineup_payload, field, value):
    with pytest.raises(ValidationError) as error:
        DailyLineupRequest(**{**daily_lineup_payload, field: value})
    assert any(item["loc"][0] == field for item in error.value.errors())

@pytest.mark.parametrize("duplicate, error", [
    ("id", "Duplicate player IDs"),
    ("name", "Duplicate player names")
])
def test_daily_lineup_request_rejects_duplicate_players(daily_lineup_payload, duplicate, error):
    original = daily_lineup_payload["players"][0]
    second = {**original, "player_id": 999999, "name": "Another Player"}
    if duplicate == "id":
        second["player_id"] = original["player_id"]
    else:
        second["name"] = f"  {original['name'].upper()}  "
    daily_lineup_payload["players"].append(second)
    with pytest.raises(ValidationError, match=error):
        DailyLineupRequest(**daily_lineup_payload)

def test_daily_lineup_request_accepts_distinct_players(daily_lineup_payload):
    second = {"player_id": 201939,"name": "Stephen Curry","positions": ["PG"],"team": "GSW","schedule": ["2026-02-05"]}
    daily_lineup_payload["players"].append(second)
    request = DailyLineupRequest(**daily_lineup_payload)
    assert [player.player_id for player in request.players] == [203999, 201939]

def test_daily_lineup_request_rejects_cutoff_after_game(daily_lineup_payload):
    with pytest.raises(ValidationError, match="as_of_date cannot be after game_date"):
        DailyLineupRequest(**{**daily_lineup_payload, "as_of_date": "2026-02-06"})

def test_daily_lineup_request_validates_nested_players(daily_lineup_payload):
    daily_lineup_payload["players"][0]["positions"] = ["INVALID"]
    with pytest.raises(ValidationError) as error:
        DailyLineupRequest(**daily_lineup_payload)
    assert any(item["loc"][:3] == ("players", 0, "positions") for item in error.value.errors())

def test_daily_lineup_request_preserves_input_and_serializes_dates(daily_lineup_payload):
    original = deepcopy(daily_lineup_payload)
    request = DailyLineupRequest(**{**daily_lineup_payload, "as_of_date": "2026-02-01"})
    serialized = request.model_dump(mode="json")
    assert serialized["game_date"] == "2026-02-05"
    assert serialized["as_of_date"] == "2026-02-01"
    assert serialized["players"][0]["schedule"] == ["2026-02-03", "2026-02-05"]
    assert daily_lineup_payload == original

@pytest.fixture
def daily_lineup_response_payload():
    return {
        "season": "2025-26",
        "game_date": "2026-02-05",
        "model": "linear_regression",
        "as_of_date": "2026-02-01",
        "lineup": {
            "PG": {
                "player_id": 201939,
                "name": "Stephen Curry",
                "positions": ["PG"],
                "team": "GSW",
                "projected_fp": 35.14},
            "C1": None},
        "projected_total": 35.14}

def test_daily_lineup_response_parses_nested_players_and_serializes(daily_lineup_response_payload):
    response = DailyLineupResponse(**daily_lineup_response_payload)
    assert response.game_date == date(2026, 2, 5)
    assert response.as_of_date == date(2026, 2, 1)
    assert isinstance(response.lineup["PG"], ProjectedPlayerResponse)
    assert response.lineup["PG"].player_id == 201939
    assert response.lineup["PG"].projected_fp == pytest.approx(35.14)
    assert response.lineup["C1"] is None
    assert response.projected_total == pytest.approx(35.14)
    assert response.model_dump(mode="json") == daily_lineup_response_payload

@pytest.mark.parametrize("model", ["linear_regression", "random_forest", "pytorch"])
def test_daily_lineup_response_accepts_supported_models(daily_lineup_response_payload, model):
    response = DailyLineupResponse(**{**daily_lineup_response_payload, "model": model})
    assert response.model == model

def test_daily_lineup_response_accepts_empty_lineup(daily_lineup_response_payload):
    response = DailyLineupResponse(**{**daily_lineup_response_payload, "lineup": {}, "projected_total": 0.0})
    assert response.lineup == {}
    assert response.projected_total == 0.0

@pytest.mark.parametrize("points", [0.0, -5.0, 42.5])
def test_daily_lineup_response_accepts_finite_projections(daily_lineup_response_payload, points):
    daily_lineup_response_payload["lineup"]["PG"]["projected_fp"] = points
    daily_lineup_response_payload["projected_total"] = points
    response = DailyLineupResponse(**daily_lineup_response_payload)
    assert response.lineup["PG"].projected_fp == points
    assert response.projected_total == points

@pytest.mark.parametrize("points", [float("nan"), float("inf"), float("-inf")])
def test_daily_lineup_response_rejects_nonfinite_total(daily_lineup_response_payload, points):
    with pytest.raises(ValidationError) as error:
        DailyLineupResponse(**{**daily_lineup_response_payload, "projected_total": points})
    assert any(item["loc"] == ("projected_total",) for item in error.value.errors())

@pytest.mark.parametrize("points", [float("nan"), float("inf"), float("-inf")])
def test_daily_lineup_response_rejects_nonfinite_player_projection(daily_lineup_response_payload, points):
    daily_lineup_response_payload["lineup"]["PG"]["projected_fp"] = points
    with pytest.raises(ValidationError) as error:
        DailyLineupResponse(**daily_lineup_response_payload)
    assert any(item["loc"] == ("lineup", "PG", "projected_fp") for item in error.value.errors())

@pytest.mark.parametrize("field", ["season", "game_date", "model", "as_of_date", "lineup", "projected_total"])
def test_daily_lineup_response_requires_all_fields(daily_lineup_response_payload, field):
    del daily_lineup_response_payload[field]
    with pytest.raises(ValidationError) as error:
        DailyLineupResponse(**daily_lineup_response_payload)
    assert any(item["loc"] == (field,) and item["type"] == "missing" for item in error.value.errors())

@pytest.mark.parametrize("field", ["player_id", "name", "positions", "team", "projected_fp"])
def test_projected_player_response_requires_all_fields(daily_lineup_response_payload, field):
    player = daily_lineup_response_payload["lineup"]["PG"].copy()
    del player[field]
    with pytest.raises(ValidationError) as error:
        ProjectedPlayerResponse(**player)
    assert any(item["loc"] == (field,) and item["type"] == "missing" for item in error.value.errors())

@pytest.mark.parametrize("field, value", [
    ("player_id", 0),
    ("positions", ["UTIL"]),
    ("positions", ["pg"])
])
def test_projected_player_response_rejects_invalid_fields(daily_lineup_response_payload, field, value):
    player = {**daily_lineup_response_payload["lineup"]["PG"], field: value}
    with pytest.raises(ValidationError) as error:
        ProjectedPlayerResponse(**player)
    assert any(item["loc"][0] == field for item in error.value.errors())

@pytest.fixture
def weekly_lineup_payload(roster_player_payload):
    return {
        "players": [deepcopy(roster_player_payload)],
        "season": "2025-26",
        "start_date": "2026-02-02",
        "end_date": "2026-02-08",
        "lineup_slots": ["C1", "UTIL1"]}

def test_weekly_lineup_request_parses_roster_and_sets_defaults(weekly_lineup_payload):
    request = WeeklyLineupRequest(**weekly_lineup_payload)
    assert isinstance(request.players[0], RosterPlayerRequest)
    assert request.start_date == date(2026, 2, 2)
    assert request.end_date == date(2026, 2, 8)
    assert request.players[0].schedule == [date(2026, 2, 3), date(2026, 2, 5)]
    assert request.lineup_slots == ["C1", "UTIL1"]
    assert request.model == "linear_regression"
    assert request.as_of_date is None

@pytest.mark.parametrize("model", ["linear_regression", "random_forest", "pytorch"])
def test_weekly_lineup_request_accepts_supported_models(weekly_lineup_payload, model):
    request = WeeklyLineupRequest(**{**weekly_lineup_payload, "model": model})
    assert request.model == model

@pytest.mark.parametrize("cutoff", ["2026-02-01", "2026-02-02"])
def test_weekly_lineup_request_accepts_valid_cutoffs(weekly_lineup_payload, cutoff):
    request = WeeklyLineupRequest(**{**weekly_lineup_payload, "as_of_date": cutoff})
    assert request.as_of_date == date.fromisoformat(cutoff)

def test_weekly_lineup_request_accepts_single_day(weekly_lineup_payload):
    request = WeeklyLineupRequest(**{**weekly_lineup_payload, "end_date": "2026-02-02"})
    assert request.start_date == request.end_date

@pytest.mark.parametrize("field", ["players", "season", "start_date", "end_date", "lineup_slots"])
def test_weekly_lineup_request_requires_fields(weekly_lineup_payload, field):
    del weekly_lineup_payload[field]
    with pytest.raises(ValidationError) as error:
        WeeklyLineupRequest(**weekly_lineup_payload)
    assert any(item["loc"] == (field,) and item["type"] == "missing" for item in error.value.errors())

@pytest.mark.parametrize("field, value", [
    ("players", []),
    ("season", "2025"),
    ("start_date", "2026-02-30"),
    ("end_date", "not-a-date"),
    ("lineup_slots", []),
    ("lineup_slots", ["PG", "PG"]),
    ("lineup_slots", ["C"]),
    ("model", "unknown"),
    ("as_of_date", "not-a-date")
])
def test_weekly_lineup_request_rejects_invalid_fields(weekly_lineup_payload, field, value):
    with pytest.raises(ValidationError) as error:
        WeeklyLineupRequest(**{**weekly_lineup_payload, field: value})
    assert any(item["loc"][0] == field for item in error.value.errors())

@pytest.mark.parametrize("field, value, message", [
    ("start_date", "2026-02-09", "start_date cannot be after end_date"),
    ("as_of_date", "2026-02-03", "as_of_date cannot be after start_date")
])
def test_weekly_lineup_request_rejects_invalid_date_order(weekly_lineup_payload, field, value, message):
    with pytest.raises(ValidationError, match=message):
        WeeklyLineupRequest(**{**weekly_lineup_payload, field: value})

@pytest.mark.parametrize("duplicate, message", [
    ("id", "Duplicate player IDs"),
    ("name", "Duplicate player names")
])
def test_weekly_lineup_request_rejects_duplicate_players(weekly_lineup_payload, duplicate, message):
    original = weekly_lineup_payload["players"][0]
    second = {**original, "player_id": 999999, "name": "Another Player"}
    if duplicate == "id":
        second["player_id"] = original["player_id"]
    else:
        second["name"] = f"  {original['name'].upper()}  "
    weekly_lineup_payload["players"].append(second)
    with pytest.raises(ValidationError, match=message):
        WeeklyLineupRequest(**weekly_lineup_payload)

def test_weekly_lineup_request_preserves_input_and_serializes_dates(weekly_lineup_payload):
    original = deepcopy(weekly_lineup_payload)
    request = WeeklyLineupRequest(**{**weekly_lineup_payload, "as_of_date": "2026-02-01"})
    serialized = request.model_dump(mode="json")
    assert serialized["start_date"] == "2026-02-02"
    assert serialized["end_date"] == "2026-02-08"
    assert serialized["as_of_date"] == "2026-02-01"
    assert serialized["players"][0]["schedule"] == ["2026-02-03", "2026-02-05"]
    assert weekly_lineup_payload == original

@pytest.fixture
def weekly_lineup_response_payload(daily_lineup_response_payload):
    first_day = deepcopy(daily_lineup_response_payload)
    second_day = {**deepcopy(first_day),"game_date": "2026-02-06","lineup": {"PG": None, "C1": None},"projected_total": 0.0}
    return {
        "season": "2025-26",
        "start_date": "2026-02-05",
        "end_date": "2026-02-06",
        "model": "linear_regression",
        "as_of_date": "2026-02-01",
        "daily_results": [first_day, second_day],
        "projected_total": 35.14}

def test_weekly_lineup_response_parses_nested_days_and_serializes(weekly_lineup_response_payload):
    response = WeeklyLineupResponse(**weekly_lineup_response_payload)
    assert response.start_date == date(2026, 2, 5)
    assert response.end_date == date(2026, 2, 6)
    assert response.as_of_date == date(2026, 2, 1)
    assert len(response.daily_results) == 2
    assert all(isinstance(day, DailyLineupResponse) for day in response.daily_results)
    assert isinstance(response.daily_results[0].lineup["PG"], ProjectedPlayerResponse)
    assert response.daily_results[0].lineup["PG"].name == "Stephen Curry"
    assert response.daily_results[1].lineup == {"PG": None, "C1": None}
    assert response.daily_results[1].projected_total == 0.0
    assert response.projected_total == pytest.approx(35.14)
    assert response.model_dump(mode="json") == weekly_lineup_response_payload

@pytest.mark.parametrize("model", ["linear_regression", "random_forest", "pytorch"])
def test_weekly_lineup_response_accepts_supported_models(weekly_lineup_response_payload, model):
    weekly_lineup_response_payload["model"] = model
    for day in weekly_lineup_response_payload["daily_results"]:
        day["model"] = model
    response = WeeklyLineupResponse(**weekly_lineup_response_payload)
    assert response.model == model

@pytest.mark.parametrize("points", [float("nan"), float("inf"), float("-inf")])
def test_weekly_lineup_response_rejects_nonfinite_total(weekly_lineup_response_payload, points):
    with pytest.raises(ValidationError) as error:
        WeeklyLineupResponse(**{**weekly_lineup_response_payload, "projected_total": points})
    assert any(item["loc"] == ("projected_total",) for item in error.value.errors())

@pytest.mark.parametrize("field", [
    "season", "start_date", "end_date", "model", "as_of_date", "daily_results", "projected_total"
])
def test_weekly_lineup_response_requires_all_fields(weekly_lineup_response_payload, field):
    del weekly_lineup_response_payload[field]
    with pytest.raises(ValidationError) as error:
        WeeklyLineupResponse(**weekly_lineup_response_payload)
    assert any(item["loc"] == (field,) and item["type"] == "missing" for item in error.value.errors())

def test_weekly_lineup_response_validates_nested_daily_total(weekly_lineup_response_payload):
    weekly_lineup_response_payload["daily_results"][0]["projected_total"] = float("nan")
    with pytest.raises(ValidationError) as error:
        WeeklyLineupResponse(**weekly_lineup_response_payload)
    assert any(item["loc"] == ("daily_results", 0, "projected_total") for item in error.value.errors())

def test_weekly_lineup_response_validates_nested_player_projection(weekly_lineup_response_payload):
    weekly_lineup_response_payload["daily_results"][0]["lineup"]["PG"]["projected_fp"] = float("inf")
    with pytest.raises(ValidationError) as error:
        WeeklyLineupResponse(**weekly_lineup_response_payload)
    assert any(item["loc"] == ("daily_results", 0, "lineup", "PG", "projected_fp") for item in error.value.errors())

def test_weekly_lineup_response_preserves_input(weekly_lineup_response_payload):
    original = deepcopy(weekly_lineup_response_payload)
    response = WeeklyLineupResponse(**weekly_lineup_response_payload)
    response.daily_results[0].lineup["PG"].positions.append("SG")
    assert weekly_lineup_response_payload == original