import pandas as pd
import pytest
import torch
from fantasy_basketball.inference import (
    build_next_game_features, 
    predict_next_game_sklearn, 
    predict_next_game_neural_network, 
    predict_stored_player_next_game, 
    project_roster_for_date,
    optimize_projected_day,
    calculate_projected_rest_days,
    optimize_projected_week)
from itertools import product
from unittest.mock import Mock, call
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from fantasy_basketball import inference
from fantasy_basketball.model_persistence import save_sklearn_model, save_neural_network
from sklearn.preprocessing import StandardScaler
from fantasy_basketball.torch_model import FantasyPointsNet
from fantasy_basketball.player_class import Player

ALL_FEATURES = [
    "FP_AVG_5",
    "FP_AVG_10",
    "REST_DAYS",
    "MIN_AVG_5",
    "MIN_AVG_10"]

@pytest.fixture
def prepared_history():
    history = pd.DataFrame({
        "GAME_DATE": pd.date_range("2025-11-01", periods=7).strftime("%Y-%m-%d"),
        "FANTASY_POINTS": [10, 20, 30, 40, 50, 60, 70],
        "MIN": [20, 30, 40, 50, 60, 70, 80],
        # These historical rolling values must not be reused.
        "FP_AVG_5": [-999.0] * 7,
        "FP_AVG_10": [-999.0] * 7,
        "MIN_AVG_5": [-999.0] * 7,
        "MIN_AVG_10": [-999.0] * 7,
    })
    return history.iloc[[6, 0, 4, 1, 5, 2, 3]].copy()

def test_next_game_features_uses_latest_completed_games(prepared_history):
    original = prepared_history.copy(deep=True)
    result = build_next_game_features(prepared_history,"2025-11-10",ALL_FEATURES)
    expected = pd.DataFrame([{
        "FP_AVG_5": 50.0,
        "FP_AVG_10": 40.0,
        "REST_DAYS": 2,
        "MIN_AVG_5": 60.0,
        "MIN_AVG_10": 50.0,
    }])
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(prepared_history, original)

def test_next_game_features_preserves_requested_order(prepared_history):
    columns = ["REST_DAYS", "FP_AVG_10", "FP_AVG_5"]
    result = build_next_game_features(prepared_history,"2025-11-10",columns)
    assert result.columns.tolist() == columns
    assert result.shape == (1, 3)
    assert result.iloc[0].tolist() == [2, 40.0, 50.0]

def test_next_game_features_supports_one_completed_game():
    history = pd.DataFrame({
        "GAME_DATE": ["2025-11-07"],
        "FANTASY_POINTS": [42.0],
        "MIN": [32.0],
    })
    result = build_next_game_features(history,"2025-11-08",ALL_FEATURES)
    assert result.iloc[0].to_dict() == {
        "FP_AVG_5": 42.0,
        "FP_AVG_10": 42.0,
        "REST_DAYS": 0,
        "MIN_AVG_5": 32.0,
        "MIN_AVG_10": 32.0}

def test_next_game_features_uses_calendar_days():
    history = pd.DataFrame({
        "GAME_DATE": ["2025-11-07 23:00:00"],
        "FANTASY_POINTS": [42.0],
        "MIN": [32.0]})
    result = build_next_game_features(
        history,
        "2025-11-09 01:00:00",
        ["REST_DAYS"])
    assert result.loc[0, "REST_DAYS"] == 1

@pytest.mark.parametrize(
    "game_date",
    ["2025-11-07", "2025-11-06"]
)
def test_next_game_features_rejects_nonfuture_date(prepared_history, game_date):
    with pytest.raises(ValueError, match="after the latest"):
        build_next_game_features(prepared_history,game_date,ALL_FEATURES)

def test_next_game_features_rejects_empty_history(prepared_history):
    with pytest.raises(ValueError, match="history cannot be empty"):
        build_next_game_features(prepared_history.iloc[:0],"2025-11-10",ALL_FEATURES)

def test_next_game_features_rejects_empty_feature_list(prepared_history):
    with pytest.raises(ValueError, match="At least one feature"):
        build_next_game_features(prepared_history,"2025-11-10",[])

def test_next_game_features_rejects_unknown_feature(prepared_history):
    with pytest.raises(ValueError, match="Unknown feature"):
        build_next_game_features(prepared_history,"2025-11-10",["FP_AVG_5", "UNKNOWN"])

def test_next_game_features_rejects_missing_history_date(prepared_history):
    prepared_history.loc[0, "GAME_DATE"] = None
    with pytest.raises(ValueError, match="missing dates"):
        build_next_game_features(prepared_history,"2025-11-10",ALL_FEATURES)

def test_next_game_features_rejects_missing_upcoming_date(prepared_history):
    with pytest.raises(ValueError, match="date cannot be missing"):
        build_next_game_features(prepared_history,"NaT",ALL_FEATURES)

@pytest.mark.parametrize("model_type", ["linear", "forest"])
def test_predict_next_game_sklearn_uses_saved_model(tmp_path, prepared_history, model_type):
    columns = ["REST_DAYS", "FP_AVG_10", "FP_AVG_5"]
    training_inputs = pd.DataFrame(list(product([0, 2], [20.0, 60.0], [10.0, 70.0])),columns=columns)
    target = (
        4 * training_inputs["REST_DAYS"]
        + 3 * training_inputs["FP_AVG_10"]
        + 2 * training_inputs["FP_AVG_5"]
        + 5)
    if model_type == "linear":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=10,random_state=42)
    model.fit(training_inputs, target)
    path = tmp_path / "model.pkl"
    save_sklearn_model(model, str(path), {})
    original = prepared_history.copy(deep=True)
    projection = predict_next_game_sklearn(prepared_history,"2025-11-10",str(path))
    # These expected features are calculated independently:
    # rest=2, last-ten FP average=40, last-five FP average=50.
    expected_features = pd.DataFrame([[2, 40.0, 50.0]],columns=columns)
    expected = float(model.predict(expected_features)[0])
    assert type(projection) is float
    assert projection == pytest.approx(expected)
    if model_type == "linear":
        assert projection == pytest.approx(233.0)
    pd.testing.assert_frame_equal(prepared_history, original)


@pytest.mark.parametrize(
    "bad_prediction",
    [float("nan"), float("inf"), float("-inf")]
)
def test_predict_next_game_sklearn_rejects_nonfinite_prediction(monkeypatch, prepared_history, bad_prediction):
    model = Mock()
    model.predict.return_value = np.array([bad_prediction])
    loader = Mock(return_value={"model": model,"feature_columns": ["FP_AVG_5", "FP_AVG_10", "REST_DAYS"]})
    monkeypatch.setattr(inference, "load_sklearn_model", loader)
    with pytest.raises(ValueError, match="Prediction must be finite"):
        predict_next_game_sklearn(prepared_history,"2025-11-10","unused.pkl")

def test_predict_next_game_sklearn_missing_model(tmp_path, prepared_history):
    with pytest.raises(FileNotFoundError):
        predict_next_game_sklearn(prepared_history,"2025-11-10",str(tmp_path / "missing.pkl"))

def test_predict_next_game_sklearn_validates_date_before_predicting(monkeypatch, prepared_history):
    model = Mock()
    loader = Mock(return_value={"model": model,"feature_columns": ["FP_AVG_5", "FP_AVG_10", "REST_DAYS"]})
    monkeypatch.setattr(inference, "load_sklearn_model", loader)
    with pytest.raises(ValueError, match="after the latest"):
        predict_next_game_sklearn(prepared_history,"2025-11-07","unused.pkl")
    model.predict.assert_not_called()

@pytest.mark.parametrize(
    "columns",
    [
        ["REST_DAYS", "FP_AVG_10", "FP_AVG_5"],
        ALL_FEATURES,
    ]
)
def test_predict_next_game_neural_network_uses_saved_bundle(tmp_path, prepared_history, columns):
    training_features = pd.DataFrame({
        "FP_AVG_5": [10.0, 30.0],
        "FP_AVG_10": [20.0, 40.0],
        "REST_DAYS": [0.0, 2.0],
        "MIN_AVG_5": [20.0, 40.0],
        "MIN_AVG_10": [22.0, 42.0]})
    scaler = StandardScaler().fit(training_features[columns])
    model = FantasyPointsNet(input_size=len(columns))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.25)
    path = tmp_path / "network.pt"
    save_neural_network(model, scaler, str(path), {})
    original = prepared_history.copy(deep=True)
    projection = predict_next_game_neural_network(prepared_history,"2025-11-10",str(path))
    # Calculate expected inputs independently of the feature builder.
    expected_features = pd.DataFrame([{
        "FP_AVG_5": 50.0,
        "FP_AVG_10": 40.0,
        "REST_DAYS": 2,
        "MIN_AVG_5": 60.0,
        "MIN_AVG_10": 50.0}])[columns]
    inputs = torch.tensor(scaler.transform(expected_features),dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        expected = float(model(inputs)[0].item())
    assert type(projection) is float
    assert projection == pytest.approx(expected)
    pd.testing.assert_frame_equal(prepared_history, original)

@pytest.mark.parametrize(
    "bad_prediction",
    [float("nan"), float("inf"), float("-inf")]
)
def test_predict_next_game_neural_network_rejects_nonfinite_prediction(monkeypatch, prepared_history, bad_prediction):
    loader = Mock(return_value={
        "model": object(),
        "scaler": object(),
        "feature_columns": ["FP_AVG_5", "FP_AVG_10", "REST_DAYS"]})
    predictor = Mock(return_value=np.array([bad_prediction]))
    monkeypatch.setattr(inference, "load_neural_network", loader)
    monkeypatch.setattr(inference, "predict_neural_network", predictor)
    with pytest.raises(ValueError, match="Prediction must be finite"):
        predict_next_game_neural_network(prepared_history,"2025-11-10","unused.pt")

def test_predict_next_game_neural_network_missing_model(tmp_path, prepared_history):
    with pytest.raises(FileNotFoundError):
        predict_next_game_neural_network(prepared_history,"2025-11-10",str(tmp_path / "missing.pt"))

@pytest.fixture(params=["sklearn", "pytorch"])
def stored_prediction_source(request, monkeypatch):
    logs = pd.DataFrame({
        "GAME_DATE": pd.to_datetime([
            "2025-11-01",
            "2025-11-03",
            "2025-11-05 00:00:00",
            "2025-11-05 23:00:00",
            "2025-11-06"],
            format="mixed"),
        "MIN": [20.0, 40.0, 999.0, 999.0, 999.0],
        "PTS": [10, 20, 999, 999, 999],
        "REB": [4, 8, 999, 999, 999],
        "AST": [0] * 5,
        "STL": [0] * 5,
        "BLK": [0] * 5,
        "TOV": [0] * 5})
    weights = {"PTS": 2.0,"REB": 0.5,"AST": 3.0,"STL": 4.0,"BLK": 5.0,"TOV": -2.0}
    columns = ["REST_DAYS","MIN_AVG_10","FP_AVG_5","MIN_AVG_5","FP_AVG_10"]
    model = Mock()
    model.predict.return_value = np.array([77.25])
    scaler = object()
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_columns": columns,
        "metadata": {"scoring_weights": weights}}
    log_fetcher = Mock(return_value=logs)
    sklearn_loader = Mock(return_value=bundle)
    neural_loader = Mock(return_value=bundle)
    neural_predictor = Mock(return_value=np.array([77.25]))
    monkeypatch.setattr(inference, "load_player_game_logs", log_fetcher)
    monkeypatch.setattr(inference, "load_sklearn_model", sklearn_loader)
    monkeypatch.setattr(inference, "load_neural_network", neural_loader)
    monkeypatch.setattr(inference, "predict_neural_network", neural_predictor)
    return {
        "model_type": request.param,
        "connection": object(),
        "logs": logs,
        "log_fetcher": log_fetcher,
        "model": model,
        "scaler": scaler,
        "sklearn_loader": sklearn_loader,
        "neural_loader": neural_loader,
        "neural_predictor": neural_predictor}

def test_stored_prediction_filters_history_and_uses_saved_weights(stored_prediction_source):
    source = stored_prediction_source
    original = source["logs"].copy(deep=True)
    projection = predict_stored_player_next_game(
        source["connection"],
        10,
        "2025-26",
        "2025-11-05 12:00:00",
        "saved_model",
        model_type=source["model_type"],
        as_of_date=None)
    assert type(projection) is float
    assert projection == pytest.approx(77.25)
    source["log_fetcher"].assert_called_once_with(source["connection"], 10, "2025-26")
    # Only November 1 and 3 should contribute.
    # FP values: 10*2 + 4*0.5 = 22; 20*2 + 8*0.5 = 44.
    expected_features = pd.DataFrame([{
        "REST_DAYS": 1,
        "MIN_AVG_10": 30.0,
        "FP_AVG_5": 33.0,
        "MIN_AVG_5": 30.0,
        "FP_AVG_10": 33.0}])
    if source["model_type"] == "sklearn":
        source["sklearn_loader"].assert_called_once_with("saved_model")
        source["neural_loader"].assert_not_called()
        source["neural_predictor"].assert_not_called()
        source["model"].predict.assert_called_once()
        actual_features = source["model"].predict.call_args.args[0]
    else:
        source["neural_loader"].assert_called_once_with("saved_model")
        source["sklearn_loader"].assert_not_called()
        source["model"].predict.assert_not_called()
        source["neural_predictor"].assert_called_once()
        args = source["neural_predictor"].call_args.args
        assert args[0] is source["model"]
        assert args[2] is source["scaler"]
        actual_features = args[1]
    pd.testing.assert_frame_equal(actual_features, expected_features)
    pd.testing.assert_frame_equal(source["logs"], original)

@pytest.mark.parametrize("case", ["empty", "same_day_or_later"])
def test_stored_prediction_rejects_missing_prior_history(stored_prediction_source, case):
    source = stored_prediction_source
    logs = source["logs"]
    if case == "empty":
        source["log_fetcher"].return_value = pd.DataFrame()
    else:
        source["log_fetcher"].return_value = logs.loc[logs["GAME_DATE"] >= pd.Timestamp("2025-11-05")].copy()
    with pytest.raises(ValueError, match="No completed games"):
        predict_stored_player_next_game(
            source["connection"],
            10,
            "2025-26",
            "2025-11-05",
            "saved_model",
            model_type=source["model_type"],
            as_of_date=None)
    source["model"].predict.assert_not_called()
    source["neural_predictor"].assert_not_called()

def test_stored_prediction_rejects_missing_game_date(stored_prediction_source):
    source = stored_prediction_source
    with pytest.raises(ValueError, match="date cannot be missing"):
        predict_stored_player_next_game(
            source["connection"],
            10,
            "2025-26",
            "NaT",
            "saved_model",
            model_type=source["model_type"],
            as_of_date=None)

def test_stored_prediction_rejects_missing_history_date(stored_prediction_source):
    source = stored_prediction_source
    source["logs"].loc[0, "GAME_DATE"] = pd.NaT
    with pytest.raises(ValueError, match="missing dates"):
        predict_stored_player_next_game(
            source["connection"],
            10,
            "2025-26",
            "2025-11-05",
            "saved_model",
            model_type=source["model_type"],
            as_of_date=None)

@pytest.mark.parametrize(
    "bad_prediction",
    [float("nan"), float("inf"), float("-inf")]
)
def test_stored_prediction_rejects_nonfinite_result(stored_prediction_source, bad_prediction):
    source = stored_prediction_source
    output = np.array([bad_prediction])
    source["model"].predict.return_value = output
    source["neural_predictor"].return_value = output
    with pytest.raises(ValueError, match="Prediction must be finite"):
        predict_stored_player_next_game(
            source["connection"],
            10,
            "2025-26",
            "2025-11-05",
            "saved_model",
            model_type=source["model_type"],
            as_of_date=None)

def test_stored_prediction_rejects_invalid_model_type():
    with pytest.raises(ValueError, match="Invalid model type"):
        predict_stored_player_next_game(None,10,"2025-26","2025-11-05","unused",model_type="unknown",as_of_date=None)

@pytest.mark.parametrize("model_type", ["sklearn", "pytorch"])
def test_project_roster_projects_only_scheduled_players(monkeypatch, model_type):
    game_date = "2026-02-01"
    players = [
        Player("First Player", ["PG", "SG"], 10.0, [game_date, "2026-02-03"], "BOS"),
        Player("Off-Day Player", ["C"], 20.0, ["2026-02-02"], "DEN"),
        Player("Second Player", ["SF", "PF"], 30.0, [game_date], "NYK")]
    player_ids = {"First Player": 101, "Second Player": 202}
    connection = object()
    predictor = Mock(side_effect=[42.5, 35.25])
    monkeypatch.setattr(inference, "predict_stored_player_next_game", predictor)
    projected = project_roster_for_date(connection, players, player_ids, "2025-26", game_date, "saved_model", model_type)
    assert predictor.call_args_list == [
        call(connection, 101, "2025-26", game_date, "saved_model", model_type, as_of_date=None, scheduled_dates=[game_date, "2026-02-03"]),
        call(connection, 202, "2025-26", game_date, "saved_model", model_type, as_of_date=None, scheduled_dates=[game_date])]
    assert len(projected) == 2
    for result, original, expected_projection in [(projected[0], players[0], 42.5),(projected[1], players[2], 35.25)]:
        assert isinstance(result, Player)
        assert result is not original
        assert result._name == original._name
        assert result._positions == original._positions
        assert result._positions is not original._positions
        assert result._team == original._team
        assert result._projected_points == expected_projection
        assert result._schedule == [game_date]
    assert [player._projected_points for player in players] == [10.0, 20.0, 30.0]
    assert players[0]._schedule == [game_date, "2026-02-03"]
    assert players[1]._schedule == ["2026-02-02"]
    assert players[2]._schedule == [game_date]
    projected[0]._positions.append("SF")
    assert players[0]._positions == ["PG", "SG"]

def test_project_roster_rejects_missing_id_for_playing_player(monkeypatch):
    game_date = "2026-02-01"
    player = Player("Missing Player", ["C"], 20.0, [game_date], "DEN")
    predictor = Mock()
    monkeypatch.setattr(inference, "predict_stored_player_next_game", predictor)
    with pytest.raises(ValueError, match="Missing NBA player ID for Missing Player"):
        project_roster_for_date(None, [player], {}, "2025-26", game_date, "unused.pkl")
    predictor.assert_not_called()

@pytest.mark.parametrize("has_off_day_player", [False, True])
def test_project_roster_returns_empty_when_nobody_plays(monkeypatch, has_off_day_player):
    players = []
    if has_off_day_player:
        players = [Player("Off-Day Player", ["PG"], 20.0, ["2026-02-02"], "BOS")]
    predictor = Mock()
    monkeypatch.setattr(inference, "predict_stored_player_next_game", predictor)
    result = project_roster_for_date(None, players, {}, "2025-26", "2026-02-01", "unused.pkl")
    assert result == []
    predictor.assert_not_called()

def test_project_roster_propagates_prediction_failure(monkeypatch):
    game_date = "2026-02-01"
    player = Player("Test Player", ["C"], 20.0, [game_date], "DEN")
    predictor = Mock(side_effect=ValueError("No completed games before prediction date"))
    monkeypatch.setattr(inference, "predict_stored_player_next_game", predictor)
    with pytest.raises(ValueError, match="No completed games"):
        project_roster_for_date(None, [player], {"Test Player": 101}, "2025-26", game_date, "unused.pkl")
    assert player._projected_points == 20.0
    assert player._schedule == [game_date]

@pytest.mark.parametrize("model_type", ["sklearn", "pytorch"])
def test_optimize_projected_day_finds_best_lineup_using_model_projections(monkeypatch, model_type):
    game_date = "2026-02-01"
    players = [
        Player("Flexible Guard", ["PG", "SG"], 1.0, [game_date], "BOS"),
        Player("Point Guard", ["PG"], 1000.0, [game_date], "NYK"),
        Player("Shooting Guard", ["SG"], 500.0, [game_date], "DEN"),
        Player("Off-Day Star", ["PG", "SG"], 9999.0, ["2026-02-02"], "CLE")]
    player_ids = {"Flexible Guard": 101, "Point Guard": 202, "Shooting Guard": 303}
    projections = {101: 50.0, 202: 40.0, 303: 30.0}
    connection = object()
    def predict_player(connection, player_id, season, game_date, model_path, model_type,as_of_date=None, scheduled_dates=None):
        return projections[player_id]
    predictor = Mock(side_effect=predict_player)
    monkeypatch.setattr(inference, "predict_stored_player_next_game", predictor)
    lineup, total = optimize_projected_day(
        connection, 
        players, 
        player_ids, 
        "2025-26", 
        game_date, 
        ["PG", "SG"], 
        "saved_model", 
        model_type)
    # The flexible guard must play SG so the stronger PG-only player can start.
    assert lineup["PG"]._name == "Point Guard"
    assert lineup["SG"]._name == "Flexible Guard"
    assert lineup["PG"]._projected_points == 40.0
    assert lineup["SG"]._projected_points == 50.0
    assert total == pytest.approx(90.0)
    assert len({id(player) for player in lineup.values()}) == 2
    assert all(player.plays_on(game_date) for player in lineup.values())
    assert predictor.call_args_list == [
        call(connection, 101, "2025-26", game_date, "saved_model", model_type, as_of_date=None, scheduled_dates=[game_date]),
        call(connection, 202, "2025-26", game_date, "saved_model", model_type, as_of_date=None, scheduled_dates=[game_date]),
        call(connection, 303, "2025-26", game_date, "saved_model", model_type, as_of_date=None, scheduled_dates=[game_date])]
    assert [player._projected_points for player in players] == [1.0, 1000.0, 500.0, 9999.0]

@pytest.mark.parametrize("empty_roster", [True, False])
def test_optimize_projected_day_returns_zero_when_nobody_plays(monkeypatch, empty_roster):
    players = []
    if not empty_roster:
        players = [Player("Off-Day Player", ["PG"], 50.0, ["2026-02-02"], "BOS")]
    predictor = Mock()
    monkeypatch.setattr(inference, "predict_stored_player_next_game", predictor)
    lineup, total = optimize_projected_day(None, players, {}, "2025-26", "2026-02-01", ["PG", "SG"], "unused.pkl")
    assert total == pytest.approx(0.0)
    assert all(player is None for player in lineup.values())
    predictor.assert_not_called()

def test_optimize_projected_day_propagates_prediction_failure(monkeypatch):
    game_date = "2026-02-01"
    players = [Player("Test Player", ["PG"], 20.0, [game_date], "BOS")]
    predictor = Mock(side_effect=ValueError("No completed games before prediction date"))
    monkeypatch.setattr(inference, "predict_stored_player_next_game", predictor)
    with pytest.raises(ValueError, match="No completed games"):
        optimize_projected_day(None, players, {"Test Player": 101}, "2025-26", game_date, ["PG"], "unused.pkl")
    assert players[0]._projected_points == 20.0

def test_stored_prediction_uses_fixed_history_cutoff(stored_prediction_source):
    source = stored_prediction_source
    original = source["logs"].copy(deep=True)
    projection = predict_stored_player_next_game(
        source["connection"],
        10,
        "2025-26",
        "2025-11-05",
        "saved_model",
        model_type=source["model_type"],
        as_of_date="2025-11-03 12:00:00")
    # The cutoff normalizes to November 3, so only November 1 contributes.
    expected = pd.DataFrame([{
        "REST_DAYS": 3,
        "MIN_AVG_10": 20.0,
        "FP_AVG_5": 22.0,
        "MIN_AVG_5": 20.0,
        "FP_AVG_10": 22.0}])
    if source["model_type"] == "sklearn":
        actual = source["model"].predict.call_args.args[0]
    else:
        actual = source["neural_predictor"].call_args.args[1]
    assert projection == pytest.approx(77.25)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(source["logs"], original)

@pytest.mark.parametrize("cutoff, error", [
    ("NaT", "History cutoff date cannot be missing"),
    ("2025-11-06", "History cutoff cannot be after prediction date"),
    ("2025-11-01", "No completed games")
])
def test_stored_prediction_rejects_invalid_or_unusable_cutoff(stored_prediction_source, cutoff, error):
    source = stored_prediction_source
    with pytest.raises(ValueError, match=error):
        predict_stored_player_next_game(
            source["connection"],
            10,
            "2025-26",
            "2025-11-05",
            "saved_model",
            model_type=source["model_type"],
            as_of_date=cutoff)
    source["model"].predict.assert_not_called()
    source["neural_predictor"].assert_not_called()

@pytest.mark.parametrize("model_type", ["sklearn", "pytorch"])
def test_daily_optimizer_passes_cutoff_through_roster_to_prediction(monkeypatch, model_type):
    game_date = "2026-02-05"
    cutoff = "2026-02-01"
    player = Player("Test Player", ["PG"], 10.0, [game_date], "BOS")
    connection = object()
    predictor = Mock(return_value=42.5)
    monkeypatch.setattr(inference, "predict_stored_player_next_game", predictor)
    lineup, total = optimize_projected_day(
        connection,
        [player],
        {"Test Player": 101},
        "2025-26",
        game_date,
        ["PG"],
        "saved_model",
        model_type,
        as_of_date=cutoff)
    predictor.assert_called_once_with(
        connection,
        101,
        "2025-26",
        game_date,
        "saved_model",
        model_type,
        as_of_date=cutoff,
        scheduled_dates=[game_date])
    assert lineup["PG"]._name == "Test Player"
    assert lineup["PG"]._projected_points == 42.5
    assert total == pytest.approx(42.5)
    assert player._projected_points == 10.0

@pytest.mark.parametrize("schedule, expected", [
    (["2026-02-01", "2026-02-03", "2026-02-05"], 1),
    (["2026-02-04", "2026-02-05"], 0),
    (["2026-02-01"], 3),
    ([], 5),
    (["2026-02-05", "2026-02-06"], 5),
    (["2026-01-31"], 5),
    (["2026-02-03", "2026-02-01", "2026-02-03"], 1)
])
def test_calculate_projected_rest_days_uses_schedule_after_cutoff(schedule, expected):
    result = calculate_projected_rest_days("2026-01-30", "2026-02-05", schedule, "2026-02-01")
    assert result == expected
    assert type(result) is int

def test_calculate_projected_rest_days_normalizes_times():
    result = calculate_projected_rest_days(
        "2026-01-30 23:00:00", "2026-02-05 01:00:00",
        ["2026-02-03 23:00:00"], "2026-02-01 12:00:00")
    assert result == 1

def test_calculate_projected_rest_days_allows_target_on_cutoff():
    result = calculate_projected_rest_days("2026-01-30", "2026-02-01", ["2026-02-01"], "2026-02-01")
    assert result == 1

@pytest.mark.parametrize("last_completed, upcoming, cutoff", [
    ("2026-02-01", "2026-02-05", "2026-02-01"),
    ("2026-02-02", "2026-02-05", "2026-02-01"),
    ("2026-01-30", "2026-02-05", "2026-02-06")
])
def test_calculate_projected_rest_days_rejects_invalid_order(last_completed, upcoming, cutoff):
    with pytest.raises(ValueError, match="Invalid rest-day date order"):
        calculate_projected_rest_days(last_completed, upcoming, [], cutoff)

@pytest.mark.parametrize("last_completed, upcoming, cutoff", [
    ("NaT", "2026-02-05", "2026-02-01"),
    ("2026-01-30", "NaT", "2026-02-01"),
    ("2026-01-30", "2026-02-05", "NaT")
])
def test_calculate_projected_rest_days_rejects_missing_dates(last_completed, upcoming, cutoff):
    with pytest.raises(ValueError, match="Rest-day dates cannot be missing"):
        calculate_projected_rest_days(last_completed, upcoming, [], cutoff)

def test_calculate_projected_rest_days_rejects_missing_scheduled_date():
    with pytest.raises(ValueError, match="Scheduled dates cannot be missing"):
        calculate_projected_rest_days("2026-01-30", "2026-02-05", ["NaT"], "2026-02-01")

def test_calculate_projected_rest_days_preserves_schedule():
    schedule = ["2026-02-03", "2026-02-01", "2026-02-03"]
    original = schedule.copy()
    calculate_projected_rest_days("2026-01-30", "2026-02-05", schedule, "2026-02-01")
    assert schedule == original

@pytest.mark.parametrize("schedule, expected_rest", [
    (None, 3),
    ([], 3),
    (["2025-11-03", "2025-11-05"], 1),
    (["2025-11-04", "2025-11-05"], 0),
    (["2025-11-05", "2025-11-06"], 3)
])
def test_stored_prediction_uses_scheduled_rest_with_frozen_history(stored_prediction_source, schedule, expected_rest):
    source = stored_prediction_source
    original = source["logs"].copy(deep=True)
    projection = predict_stored_player_next_game(
        source["connection"],
        10,
        "2025-26",
        "2025-11-05",
        "saved_model",
        model_type=source["model_type"],
        as_of_date="2025-11-03",
        scheduled_dates=schedule)
    # Only November 1 supplies performance data, regardless of the schedule.
    expected = pd.DataFrame([{
        "REST_DAYS": expected_rest,
        "MIN_AVG_10": 20.0,
        "FP_AVG_5": 22.0,
        "MIN_AVG_5": 20.0,
        "FP_AVG_10": 22.0}])
    if source["model_type"] == "sklearn":
        actual = source["model"].predict.call_args.args[0]
    else:
        actual = source["neural_predictor"].call_args.args[1]
    assert projection == pytest.approx(77.25)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(source["logs"], original)

@pytest.mark.parametrize("model_type", ["sklearn", "pytorch"])
def test_daily_optimizer_passes_player_schedule_to_prediction(monkeypatch, model_type):
    game_date = "2026-02-05"
    cutoff = "2026-02-01"
    schedule = ["2026-02-01", "2026-02-03", game_date]
    player = Player("Test Player", ["PG"], 10.0, schedule.copy(), "BOS")
    connection = object()
    predictor = Mock(return_value=42.5)
    monkeypatch.setattr(inference, "predict_stored_player_next_game", predictor)
    lineup, total = optimize_projected_day(
        connection,
        [player],
        {"Test Player": 101},
        "2025-26",
        game_date,
        ["PG"],
        "saved_model",
        model_type,
        as_of_date=cutoff)
    predictor.assert_called_once_with(
        connection,
        101,
        "2025-26",
        game_date,
        "saved_model",
        model_type,
        as_of_date=cutoff,
        scheduled_dates=schedule)
    assert lineup["PG"]._projected_points == 42.5
    assert total == pytest.approx(42.5)
    assert player._schedule == schedule
    assert player._projected_points == 10.0

@pytest.mark.parametrize("model_type", ["sklearn", "pytorch"])
@pytest.mark.parametrize("cutoff", [None, "2026-01-30"])
def test_optimize_projected_week_uses_daily_projections_and_fixed_cutoff(monkeypatch, model_type, cutoff):
    schedule = ["2026-02-01", "2026-02-03"]
    players = [
        Player("First Player", ["PG"], 999.0, schedule.copy(), "BOS"),
        Player("Second Player", ["PG"], 888.0, schedule.copy(), "NYK")]
    player_ids = {"First Player": 101, "Second Player": 202}
    connection = object()
    projections = {
        ("2026-02-01", 101): 40.0,
        ("2026-02-01", 202): 30.0,
        ("2026-02-03", 101): 20.0,
        ("2026-02-03", 202): 50.0}
    def predict_player(connection, player_id, season, game_date, model_path, model_type, as_of_date=None, scheduled_dates=None):
        return projections[(game_date, player_id)]
    predictor = Mock(side_effect=predict_player)
    monkeypatch.setattr(inference, "predict_stored_player_next_game", predictor)
    results, total = optimize_projected_week(
        connection,
        players,
        player_ids,
        "2025-26",
        "2026-02-01",
        "2026-02-03",
        ["PG"],
        "saved_model",
        model_type,
        as_of_date=cutoff)
    expected_cutoff = cutoff if cutoff is not None else "2026-02-01"
    assert list(results) == ["2026-02-01", "2026-02-02", "2026-02-03"]
    assert results["2026-02-01"]["lineup"]["PG"]._name == "First Player"
    assert results["2026-02-01"]["projected_points"] == pytest.approx(40.0)
    assert results["2026-02-02"]["projected_points"] == pytest.approx(0.0)
    assert results["2026-02-03"]["lineup"]["PG"]._name == "Second Player"
    assert results["2026-02-03"]["projected_points"] == pytest.approx(50.0)
    assert total == pytest.approx(90.0)
    assert predictor.call_args_list == [
        call(connection, 101, "2025-26", "2026-02-01", "saved_model", model_type, as_of_date=expected_cutoff, scheduled_dates=schedule),
        call(connection, 202, "2025-26", "2026-02-01", "saved_model", model_type, as_of_date=expected_cutoff, scheduled_dates=schedule),
        call(connection, 101, "2025-26", "2026-02-03", "saved_model", model_type, as_of_date=expected_cutoff, scheduled_dates=schedule),
        call(connection, 202, "2025-26", "2026-02-03", "saved_model", model_type, as_of_date=expected_cutoff, scheduled_dates=schedule)]
    assert players[0]._projected_points == 999.0
    assert players[1]._projected_points == 888.0
    assert players[0]._schedule == schedule
    assert players[1]._schedule == schedule

def test_optimize_projected_week_normalizes_dates_and_supports_single_day(monkeypatch):
    game_date = "2026-02-01"
    player = Player("Test Player", ["PG"], 10.0, [game_date], "BOS")
    predictor = Mock(return_value=42.5)
    monkeypatch.setattr(inference, "predict_stored_player_next_game", predictor)
    results, total = optimize_projected_week(
        None,
        [player],
        {"Test Player": 101},
        "2025-26",
        "2026-02-01 18:00:00",
        "2026-02-01 01:00:00",
        ["PG"],
        "saved_model",
        as_of_date="2026-02-01 23:00:00")
    assert list(results) == [game_date]
    assert total == pytest.approx(42.5)
    predictor.assert_called_once_with(
        None,
        101,
        "2025-26",
        game_date,
        "saved_model",
        "sklearn",
        as_of_date=game_date,
        scheduled_dates=[game_date])

def test_optimize_projected_week_empty_roster_returns_all_dates(monkeypatch):
    predictor = Mock()
    monkeypatch.setattr(inference, "predict_stored_player_next_game", predictor)
    results, total = optimize_projected_week(None, [], {}, "2025-26", "2026-02-01", "2026-02-03", ["PG"], "unused.pkl")
    assert list(results) == ["2026-02-01", "2026-02-02", "2026-02-03"]
    assert all(result["projected_points"] == 0.0 for result in results.values())
    assert total == 0.0
    predictor.assert_not_called()

@pytest.mark.parametrize("start, end, cutoff, model_type, error", [
    ("NaT", "2026-02-03", None, "sklearn", "Invalid date"),
    ("2026-02-01", "NaT", None, "sklearn", "Invalid date"),
    ("2026-02-03", "2026-02-01", None, "sklearn", "Invalid date range"),
    ("2026-02-01", "2026-02-03", "NaT", "sklearn", "Invalid cutoff date"),
    ("2026-02-01", "2026-02-03", "2026-02-02", "sklearn", "History cutoff cannot be after start date"),
    ("2026-02-01", "2026-02-03", None, "unknown", "Invalid model")
])
def test_optimize_projected_week_rejects_invalid_inputs_before_optimization(monkeypatch, start, end, cutoff, model_type, error):
    optimizer = Mock()
    monkeypatch.setattr(inference, "optimize_projected_day", optimizer)
    with pytest.raises(ValueError, match=error):
        optimize_projected_week(None, [], {}, "2025-26", start, end, ["PG"], "unused", model_type, as_of_date=cutoff)
    optimizer.assert_not_called()

def test_optimize_projected_week_propagates_prediction_failure(monkeypatch):
    player = Player("Test Player", ["PG"], 10.0, ["2026-02-01"], "BOS")
    predictor = Mock(side_effect=ValueError("No completed games before prediction date"))
    monkeypatch.setattr(inference, "predict_stored_player_next_game", predictor)
    with pytest.raises(ValueError, match="No completed games"):
        optimize_projected_week(
            None, [player], {"Test Player": 101}, "2025-26",
            "2026-02-01", "2026-02-03", ["PG"], "unused.pkl")
    assert player._projected_points == 10.0