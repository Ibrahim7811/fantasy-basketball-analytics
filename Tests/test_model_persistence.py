import pickle
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from fantasy_basketball.model_persistence import save_sklearn_model, load_sklearn_model, save_neural_network, load_neural_network
from fantasy_basketball.torch_model import FantasyPointsNet, predict_neural_network
from sklearn.preprocessing import StandardScaler

@pytest.fixture
def persistence_data():
    return pd.DataFrame({
        "FP_AVG_5": [10.0, 20.0, 30.0, 40.0],
        "FP_AVG_10": [12.0, 18.0, 28.0, 36.0],
        "REST_DAYS": [0.0, 1.0, 2.0, 1.0],
        "FANTASY_POINTS": [15.0, 25.0, 35.0, 45.0],
    })

@pytest.mark.parametrize("model_type", ["linear", "forest"])
def test_save_sklearn_model_preserves_bundle(tmp_path, persistence_data, model_type):
    columns = ["REST_DAYS", "FP_AVG_10", "FP_AVG_5"]
    inputs = persistence_data[columns]
    target = persistence_data["FANTASY_POINTS"]
    if model_type == "linear":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=10,random_state=42)
    model.fit(inputs, target)
    metadata = {
        "scoring_weights": {"PTS": 1.0, "REB": 1.2},
        "training_end": "2025-11-23",
        "validation_mae": 8.8247,
    }
    path = tmp_path / "nested" / "models" / "model.pkl"
    result = save_sklearn_model(model, str(path), metadata)
    assert result is None
    assert path.is_file()
    with path.open("rb") as f:
        bundle = pickle.load(f)
    assert isinstance(bundle["model"], type(model))
    assert bundle["feature_columns"] == columns
    assert bundle["metadata"] == metadata
    np.testing.assert_allclose(bundle["model"].predict(inputs),model.predict(inputs))

def test_save_sklearn_model_replaces_existing_bundle(tmp_path, persistence_data):
    inputs = persistence_data[["FP_AVG_5"]]
    target = persistence_data["FANTASY_POINTS"]
    path = tmp_path / "model.pkl"
    first_model = LinearRegression().fit(inputs, target)
    save_sklearn_model(first_model,str(path),{"version": 1})
    second_model = LinearRegression().fit(inputs, target + 100.0)
    save_sklearn_model(second_model,str(path),{"version": 2})
    with path.open("rb") as f:
        bundle = pickle.load(f)
    assert bundle["metadata"] == {"version": 2}
    assert bundle["feature_columns"] == ["FP_AVG_5"]
    np.testing.assert_allclose(bundle["model"].predict(inputs),second_model.predict(inputs))

@pytest.mark.parametrize("model_type", ["linear", "forest"])
def test_load_sklearn_model_restores_predictions(tmp_path, persistence_data, model_type):
    columns = ["REST_DAYS", "FP_AVG_10", "FP_AVG_5"]
    inputs = persistence_data[columns]
    target = persistence_data["FANTASY_POINTS"]
    if model_type == "linear":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=10,random_state=42)
    model.fit(inputs, target)
    metadata = {"scoring_weights": {"PTS": 1.0},"training_end": "2025-11-23",}
    path = tmp_path / "model.pkl"
    save_sklearn_model(model, str(path), metadata)
    bundle = load_sklearn_model(str(path))
    assert isinstance(bundle["model"], type(model))
    assert bundle["feature_columns"] == columns
    assert bundle["metadata"] == metadata
    # Predict on new rows using the restored feature order.
    future = pd.DataFrame({
        "FP_AVG_5": [25.0, 35.0],
        "REST_DAYS": [1.0, 0.0],
        "FP_AVG_10": [23.0, 32.0],
    })
    np.testing.assert_allclose(bundle["model"].predict(future[bundle["feature_columns"]]),model.predict(future[columns]))

def test_load_sklearn_model_missing_file(tmp_path):
    path = tmp_path / "missing.pkl"
    with pytest.raises(FileNotFoundError):
        load_sklearn_model(str(path))

def test_load_sklearn_model_invalid_pickle(tmp_path):
    path = tmp_path / "invalid.pkl"
    path.write_bytes(b"not a pickle")
    with pytest.raises(pickle.UnpicklingError):
        load_sklearn_model(str(path))

@pytest.mark.parametrize(
    "columns",
    [
        ["REST_DAYS", "FP_AVG_10", "FP_AVG_5"],
        ["FP_AVG_5"],
    ]
)
def test_save_neural_network_preserves_bundle(tmp_path, persistence_data, columns):
    scaler = StandardScaler().fit(persistence_data[columns])
    model = FantasyPointsNet(input_size=len(columns))
    # Use known weights so restoring a new random model cannot pass.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.25)
    metadata = {"scoring_weights": {"PTS": 1.0},"training_end": "2025-11-23"}
    path = tmp_path / "nested" / "models" / "network.pt"
    result = save_neural_network(model, scaler, str(path), metadata)
    assert result is None
    assert path.is_file()
    bundle = torch.load(path,map_location="cpu",weights_only=False)
    assert bundle["input_size"] == len(columns)
    assert bundle["feature_columns"] == columns
    assert bundle["metadata"] == metadata
    assert isinstance(bundle["scaler"], StandardScaler)
    assert bundle["scaler"].feature_names_in_.tolist() == columns
    for attribute in ("mean_", "scale_", "var_"):
        np.testing.assert_array_equal(getattr(bundle["scaler"], attribute),getattr(scaler, attribute))
    restored_model = FantasyPointsNet(input_size=bundle["input_size"])
    restored_model.load_state_dict(bundle["model_state_dict"])
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(restored_model.state_dict()[name],parameter,rtol=0,atol=0)
    future = persistence_data.drop(columns="FANTASY_POINTS").copy()
    future[columns] = future[columns] + 5.0
    np.testing.assert_allclose(
        predict_neural_network(restored_model, future, bundle["scaler"]),
        predict_neural_network(model, future, scaler))

@pytest.mark.parametrize(
    "columns",
    [
        ["REST_DAYS", "FP_AVG_10", "FP_AVG_5"],
        ["FP_AVG_5"],
    ],
)
def test_load_neural_network_restores_predictions(tmp_path, persistence_data, columns):
    scaler = StandardScaler().fit(persistence_data[columns])
    model = FantasyPointsNet(input_size=len(columns))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.25)
    metadata = {"scoring_weights": {"PTS": 1.0},"training_end": "2025-11-23"}
    path = tmp_path / "network.pt"
    save_neural_network(model, scaler, str(path), metadata)
    bundle = load_neural_network(str(path))
    restored_model = bundle["model"]
    assert isinstance(restored_model, FantasyPointsNet)
    assert restored_model.training is False
    assert restored_model.network[0].in_features == len(columns)
    assert next(restored_model.parameters()).device.type == "cpu"
    assert bundle["feature_columns"] == columns
    assert bundle["metadata"] == metadata
    assert bundle["scaler"].feature_names_in_.tolist() == columns
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(restored_model.state_dict()[name],parameter,rtol=0,atol=0)
    # New inputs, with columns in a different order and no target.
    future = persistence_data.drop(columns="FANTASY_POINTS").copy()
    future[columns] = future[columns] + 5.0
    future = future[list(reversed(future.columns))]
    np.testing.assert_allclose(
        predict_neural_network(restored_model, future, bundle["scaler"]),
        predict_neural_network(model, future, scaler),)

def test_load_neural_network_missing_file(tmp_path):
    path = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError):
        load_neural_network(str(path))