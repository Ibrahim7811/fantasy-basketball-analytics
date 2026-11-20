from itertools import product
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from fantasy_basketball.models import train_linear_regression, train_random_forest
from fantasy_basketball.training_data import FEATURE_COLUMNS
from sklearn.ensemble import RandomForestRegressor

@pytest.fixture
def training_df():
    df = pd.DataFrame(list(product([10.0, 20.0], [12.0, 24.0], [0.0, 2.0])),columns=FEATURE_COLUMNS,)

    # A known relationship lets us check predictions independently.
    df["FANTASY_POINTS"] = (
        2 * df["FP_AVG_5"]
        + 0.5 * df["FP_AVG_10"]
        - 3 * df["REST_DAYS"]
        + 4)
    df["PLAYER_ID"] = 10
    df["GAME_DATE"] = pd.date_range("2025-11-01", periods=len(df))
    return df

def test_train_linear_regression_learns_relationship(training_df):
    model = train_linear_regression(training_df)
    unseen_features = pd.DataFrame(
        {
            "FP_AVG_5": [15.0, 18.0],
            "FP_AVG_10": [18.0, 20.0],
            "REST_DAYS": [1.0, 0.0],
        })
    predictions = model.predict(unseen_features)
    assert isinstance(model, LinearRegression)
    assert predictions == pytest.approx([40.0, 50.0])

def test_train_linear_regression_uses_only_pregame_features(training_df):
    model = train_linear_regression(training_df)
    assert list(model.feature_names_in_) == FEATURE_COLUMNS
    assert model.n_features_in_ == 3

def test_train_linear_regression_rejects_empty_input(training_df):
    empty_df = training_df.iloc[:0].copy()
    with pytest.raises(ValueError, match="Training data cannot be empty"):
        train_linear_regression(empty_df)

def test_train_linear_regression_preserves_input(training_df):
    original = training_df.copy(deep=True)
    train_linear_regression(training_df)
    pd.testing.assert_frame_equal(training_df, original)

@pytest.fixture
def forest_training_df():
    # Two clearly separated groups with enough rows for leaf size 5.
    return pd.DataFrame({
            "FP_AVG_5": [10.0] * 60 + [50.0] * 60,
            "FP_AVG_10": [15.0] * 60 + [55.0] * 60,
            "REST_DAYS": [1.0] * 120,
            "FANTASY_POINTS": [20.0] * 60 + [80.0] * 60,
            "PLAYER_ID": [10] * 60 + [20] * 60,
        })

def test_train_random_forest_learns_signal(forest_training_df):
    model = train_random_forest(forest_training_df)
    features = pd.DataFrame({
            "FP_AVG_5": [10.0, 50.0],
            "FP_AVG_10": [15.0, 55.0],
            "REST_DAYS": [1.0, 1.0],
        })
    predictions = model.predict(features)
    assert isinstance(model, RandomForestRegressor)
    assert predictions == pytest.approx([20.0, 80.0])

def test_train_random_forest_uses_only_pregame_features(forest_training_df,):
    model = train_random_forest(forest_training_df)
    assert list(model.feature_names_in_) == FEATURE_COLUMNS
    assert model.n_features_in_ == 3

def test_train_random_forest_is_reproducible(forest_training_df):
    first = train_random_forest(forest_training_df, random_state=7)
    second = train_random_forest(forest_training_df, random_state=7)
    features = forest_training_df[FEATURE_COLUMNS]
    assert first.random_state == 7
    assert second.random_state == 7
    assert first.predict(features) == pytest.approx(second.predict(features))

def test_train_random_forest_rejects_empty_input(forest_training_df):
    empty_df = forest_training_df.iloc[:0].copy()
    with pytest.raises(ValueError, match="Training data cannot be empty"):
        train_random_forest(empty_df)

def test_train_random_forest_preserves_input(forest_training_df):
    original = forest_training_df.copy(deep=True)
    train_random_forest(forest_training_df)
    pd.testing.assert_frame_equal(forest_training_df, original)