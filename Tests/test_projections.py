import pandas as pd
import pytest
from fantasy_basketball.projections import predict_next_game

@pytest.fixture
def scored_logs():
    return pd.DataFrame({
        "FANTASY_POINTS": [10, 20, 30, 40, 50, 60],
    })

def test_default_prediction_window(scored_logs):
    result = predict_next_game(scored_logs)

    # Last five scores: 20, 30, 40, 50, 60
    assert result == pytest.approx(40.0)
    assert type(result) is float

@pytest.mark.parametrize(
    "window, expected",
    [
        (1, 60.0),
        (3, 50.0),
        (10, 35.0),  # Only six games available
    ],
)
def test_prediction_windows(scored_logs, window, expected):
    result = predict_next_game(scored_logs, window=window)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize("window", [0, -1])
def test_rejects_invalid_window(scored_logs, window):
    with pytest.raises(ValueError):
        predict_next_game(scored_logs, window=window)

def test_rejects_empty_game_logs():
    df = pd.DataFrame({"FANTASY_POINTS": []})

    with pytest.raises(ValueError):
        predict_next_game(df)

def test_prediction_preserves_original(scored_logs):
    original = scored_logs.copy(deep=True)

    predict_next_game(scored_logs)

    pd.testing.assert_frame_equal(scored_logs, original)