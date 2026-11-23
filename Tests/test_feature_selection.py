import numpy as np
import pandas as pd
import pytest
import torch
from fantasy_basketball.models import train_linear_regression, train_random_forest
from fantasy_basketball.torch_model import FantasyPointsNet, fit_feature_scaler, prepare_tensors, predict_neural_network, train_neural_network
from fantasy_basketball.training_data import FEATURE_COLUMNS

BASE_FEATURES = ["REST_DAYS", "FP_AVG_10", "FP_AVG_5"]

@pytest.fixture
def feature_selection_data():
    return pd.DataFrame({
        "FP_AVG_5": [10.0, 20.0, 30.0, 40.0],
        "FP_AVG_10": [12.0, 18.0, 28.0, 36.0],
        "REST_DAYS": [0.0, 1.0, 2.0, 1.0],
        "MIN_AVG_5": [20.0, 25.0, 30.0, 35.0],
        "MIN_AVG_10": [22.0, 24.0, 28.0, 32.0],
        "FANTASY_POINTS": [15.0, 25.0, 35.0, 45.0],
    })

@pytest.mark.parametrize(
    "trainer",
    [train_linear_regression, train_random_forest]
)
@pytest.mark.parametrize("columns", [None, BASE_FEATURES])
def test_sklearn_training_selects_features(feature_selection_data, trainer, columns):
    expected = FEATURE_COLUMNS if columns is None else columns
    dataset = feature_selection_data[[*expected, "FANTASY_POINTS"]].copy()
    model = trainer(dataset, feature_columns=columns)
    assert model.n_features_in_ == len(expected)
    assert model.feature_names_in_.tolist() == expected
    predictions = model.predict(dataset[expected])
    assert predictions.shape == (len(dataset),)
    assert np.isfinite(predictions).all()

@pytest.mark.parametrize(
    "trainer",
    [train_linear_regression, train_random_forest]
)
def test_sklearn_training_rejects_empty_features(feature_selection_data, trainer):
    with pytest.raises(ValueError, match="At least one feature"):
        trainer(feature_selection_data, feature_columns=[])

@pytest.mark.parametrize("input_size", [3, 5])
def test_network_accepts_selected_input_size(input_size):
    model = FantasyPointsNet(input_size=input_size)
    predictions = model(torch.zeros((4, input_size)))
    assert predictions.shape == (4,)

@pytest.mark.parametrize("input_size", [0, -1, 1.5, True])
def test_network_rejects_invalid_input_size(input_size):
    with pytest.raises(ValueError, match="input_size"):
        FantasyPointsNet(input_size=input_size)

def test_tensor_helpers_use_scaler_feature_order(feature_selection_data):
    scaler = fit_feature_scaler(feature_selection_data,feature_columns=BASE_FEATURES)
    # Reverse the DataFrame columns to check selection by name.
    shuffled = feature_selection_data[list(reversed(feature_selection_data.columns))]
    inputs, targets = prepare_tensors(shuffled, scaler)
    assert scaler.feature_names_in_.tolist() == BASE_FEATURES
    assert inputs.shape == (4, 3)
    np.testing.assert_allclose(inputs.numpy(),scaler.transform(feature_selection_data[BASE_FEATURES]),rtol=1e-6,atol=1e-6)
    np.testing.assert_array_equal(targets.numpy(),feature_selection_data["FANTASY_POINTS"].to_numpy(dtype="float32"))
    model = FantasyPointsNet(input_size=3)
    predictions = predict_neural_network(model,shuffled.drop(columns="FANTASY_POINTS"),scaler)
    with torch.no_grad():
        expected = model(inputs).numpy()
    np.testing.assert_allclose(predictions, expected)

def test_scaler_rejects_empty_features(feature_selection_data):
    with pytest.raises(ValueError, match="At least one feature"):
        fit_feature_scaler(feature_selection_data,feature_columns=[])

@pytest.mark.parametrize("columns", [None, BASE_FEATURES])
def test_neural_training_selects_features(feature_selection_data, columns):
    expected = FEATURE_COLUMNS if columns is None else columns
    train_df = feature_selection_data[[*expected, "FANTASY_POINTS"]].copy()
    validation_df = train_df.copy()
    validation_df[expected] = validation_df[expected] + 10.0
    model, scaler, history = train_neural_network(train_df,validation_df,epochs=2,patience=3,feature_columns=columns)
    assert model.network[0].in_features == len(expected)
    assert scaler.feature_names_in_.tolist() == expected
    assert len(history["train_loss"]) == 2
    assert len(history["validation_loss"]) == 2
    # Validation data must not affect the fitted scaler.
    np.testing.assert_allclose(scaler.mean_,train_df[expected].mean().to_numpy())
    predictions = predict_neural_network(model,validation_df.drop(columns="FANTASY_POINTS"),scaler)
    assert predictions.shape == (len(validation_df),)
    assert np.isfinite(predictions).all()

def test_neural_training_rejects_empty_features(feature_selection_data):
    with pytest.raises(ValueError, match="At least one feature"):
        train_neural_network(feature_selection_data,feature_selection_data,epochs=2,feature_columns=[])