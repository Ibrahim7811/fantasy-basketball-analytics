from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from fantasy_basketball import training
from fantasy_basketball.model_persistence import load_sklearn_model, load_neural_network
from fantasy_basketball.torch_model import train_neural_network, predict_neural_network

@pytest.fixture
def training_frames():
    train_df = pd.DataFrame({
        "FP_AVG_5": [10.0, 20.0, 30.0, 40.0],
        "FP_AVG_10": [12.0, 18.0, 28.0, 36.0],
        "REST_DAYS": [0.0, 1.0, 2.0, 1.0],
        "MIN_AVG_5": [20.0, 25.0, 30.0, 35.0],
        "MIN_AVG_10": [22.0, 24.0, 28.0, 32.0],
        "FANTASY_POINTS": [15.0, 25.0, 35.0, 45.0],
    })
    validation_df = pd.DataFrame({
        "FP_AVG_5": [25.0, 35.0],
        "FP_AVG_10": [23.0, 32.0],
        "REST_DAYS": [1.0, 0.0],
        "MIN_AVG_5": [27.0, 33.0],
        "MIN_AVG_10": [26.0, 30.0],
        "FANTASY_POINTS": [30.0, 40.0],
    })
    return train_df, validation_df

@pytest.mark.parametrize(
    "columns",
    [
        ["REST_DAYS", "FP_AVG_10", "FP_AVG_5"],
        [
            "FP_AVG_5",
            "FP_AVG_10",
            "REST_DAYS",
            "MIN_AVG_5",
            "MIN_AVG_10",
        ]])
def test_train_and_save_models_creates_usable_models(tmp_path, monkeypatch, training_frames, columns):
    train_df, validation_df = training_frames
    original_train = train_df.copy(deep=True)
    original_validation = validation_df.copy(deep=True)
    metadata = {"scoring_weights": {"PTS": 1.0, "REB": 1.2},"training_end": "2025-11-23"}
    output_dir = tmp_path / "nested" / "models"
    captured = {}
    def quick_neural_training(train, validation, **kwargs):
        # Keep real training, scaling, saving, and loading.
        settings = {**kwargs, "epochs": 2}
        model, scaler, history = train_neural_network(train, validation, **settings)
        captured["model"] = model
        captured["scaler"] = scaler
        return model, scaler, history
    neural_trainer = Mock(side_effect=quick_neural_training)
    monkeypatch.setattr(training, "train_neural_network", neural_trainer)
    paths = training.train_and_save_models(train_df,validation_df,str(output_dir),columns,metadata)
    assert paths == {
        "linear_regression": str(output_dir / "linear_regression.pkl"),
        "random_forest": str(output_dir / "random_forest.pkl"),
        "neural_network": str(output_dir / "neural_network.pt")}
    assert all(Path(path).is_file() for path in paths.values())
    neural_trainer.assert_called_once_with(
        train_df,
        validation_df,
        epochs=3000,
        learning_rate=0.001,
        patience=30,
        random_state=42,
        feature_columns=columns)
    # Reload both scikit-learn models and make predictions.
    for name, model_type in [("linear_regression", LinearRegression),("random_forest", RandomForestRegressor)]:
        bundle = load_sklearn_model(paths[name])
        model = bundle["model"]
        assert isinstance(model, model_type)
        assert bundle["feature_columns"] == columns
        assert bundle["metadata"] == metadata
        assert model.feature_names_in_.tolist() == columns
        predictions = model.predict(validation_df[columns])
        assert predictions.shape == (len(validation_df),)
        assert np.isfinite(predictions).all()
        if name == "random_forest":
            assert model.random_state == 42
    # Reload the neural model and compare against its original.
    bundle = load_neural_network(paths["neural_network"])
    assert bundle["feature_columns"] == columns
    assert bundle["metadata"] == metadata
    assert bundle["model"].training is False
    assert bundle["model"].network[0].in_features == len(columns)
    np.testing.assert_allclose(bundle["scaler"].mean_,train_df[columns].mean().to_numpy())
    prediction_data = validation_df.drop(columns="FANTASY_POINTS")
    np.testing.assert_allclose(
        predict_neural_network(bundle["model"],prediction_data,bundle["scaler"],),
        predict_neural_network(captured["model"],prediction_data,captured["scaler"],))
    pd.testing.assert_frame_equal(train_df, original_train)
    pd.testing.assert_frame_equal(validation_df, original_validation)