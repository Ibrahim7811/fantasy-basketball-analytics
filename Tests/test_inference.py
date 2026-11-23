import pandas as pd
import pytest
import torch
from fantasy_basketball.inference import build_next_game_features, predict_next_game_sklearn, predict_next_game_neural_network, predict_stored_player_next_game
from itertools import product
from unittest.mock import Mock
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from fantasy_basketball import inference
from fantasy_basketball.model_persistence import save_sklearn_model, save_neural_network
from sklearn.preprocessing import StandardScaler
from fantasy_basketball.torch_model import FantasyPointsNet

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
            "2025-11-06",
        ], format="mixed"),
        "MIN": [20.0, 40.0, 999.0, 999.0, 999.0],
        "PTS": [10, 20, 999, 999, 999],
        "REB": [4, 8, 999, 999, 999],
        "AST": [0] * 5,
        "STL": [0] * 5,
        "BLK": [0] * 5,
        "TOV": [0] * 5,
    })
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
    projection = predict_stored_player_next_game(source["connection"],10,"2025-26","2025-11-05 12:00:00","saved_model",model_type=source["model_type"])
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
        predict_stored_player_next_game(source["connection"],10,"2025-26","2025-11-05","saved_model",model_type=source["model_type"])
    source["model"].predict.assert_not_called()
    source["neural_predictor"].assert_not_called()

def test_stored_prediction_rejects_missing_game_date(stored_prediction_source):
    source = stored_prediction_source
    with pytest.raises(ValueError, match="date cannot be missing"):
        predict_stored_player_next_game(source["connection"],10,"2025-26","NaT","saved_model",model_type=source["model_type"])

def test_stored_prediction_rejects_missing_history_date(stored_prediction_source):
    source = stored_prediction_source
    source["logs"].loc[0, "GAME_DATE"] = pd.NaT
    with pytest.raises(ValueError, match="missing dates"):
        predict_stored_player_next_game(source["connection"],10,"2025-26","2025-11-05","saved_model",model_type=source["model_type"])

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
        predict_stored_player_next_game(source["connection"],10,"2025-26","2025-11-05","saved_model",model_type=source["model_type"])

def test_stored_prediction_rejects_invalid_model_type():
    with pytest.raises(ValueError, match="Invalid model type"):
        predict_stored_player_next_game(None,10,"2025-26","2025-11-05","unused",model_type="unknown")