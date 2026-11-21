import pandas as pd
import pytest
from fantasy_basketball.data_processing import add_minutes_features

@pytest.fixture
def minutes_logs():
    return pd.DataFrame({
            "GAME_DATE": pd.date_range("2025-11-01", periods=7),
            "MIN": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        })

def test_minutes_features_use_previous_games(minutes_logs):
    result = add_minutes_features(minutes_logs)
    expected = pd.DataFrame({
            "MIN_AVG_5": [
                float("nan"), 10.0, 15.0, 20.0, 25.0, 30.0, 40.0
            ],
            "MIN_AVG_10": [
                float("nan"), 10.0, 15.0, 20.0, 25.0, 30.0, 35.0
            ],
        })
    pd.testing.assert_frame_equal(result[["MIN_AVG_5", "MIN_AVG_10"]],expected)

def test_current_minutes_do_not_change_current_features(minutes_logs):
    before = add_minutes_features(minutes_logs)
    changed_logs = minutes_logs.copy(deep=True)
    changed_logs.loc[3, "MIN"] = 400.0
    after = add_minutes_features(changed_logs)
    columns = ["MIN_AVG_5", "MIN_AVG_10"]
    # The changed game's features and all earlier features stay unchanged.
    pd.testing.assert_frame_equal(before.loc[:3, columns],after.loc[:3, columns])
    # Later games can use the changed game's completed minutes.
    assert after.loc[4, "MIN_AVG_5"] == pytest.approx(115.0)
    assert before.loc[4, "MIN_AVG_5"] == pytest.approx(25.0)

def test_minutes_features_support_custom_window(minutes_logs):
    result = add_minutes_features(minutes_logs, windows=(2,))
    expected = pd.Series([float("nan"), 10.0, 15.0, 25.0, 35.0, 45.0, 55.0],name="MIN_AVG_2")
    pd.testing.assert_series_equal(result["MIN_AVG_2"], expected)
    assert "MIN_AVG_5" not in result.columns
    assert "MIN_AVG_10" not in result.columns

def test_minutes_features_preserve_input(minutes_logs):
    original = minutes_logs.copy(deep=True)
    result = add_minutes_features(minutes_logs)
    pd.testing.assert_frame_equal(minutes_logs, original)
    pd.testing.assert_frame_equal(result[original.columns], original)