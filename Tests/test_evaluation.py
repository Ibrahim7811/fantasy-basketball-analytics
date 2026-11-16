import pandas as pd
import pytest
from fantasy_basketball.evaluation import evaluate_baseline

@pytest.fixture
def evaluation_logs():
    return pd.DataFrame({
        "FANTASY_POINTS": [40, 30, 20],
        "FP_AVG_5": [35, 40, 20],
    })

def test_calculates_mae(evaluation_logs):
    result = evaluate_baseline(evaluation_logs)

    # Absolute errors: 5, 10, 0 → average = 5
    assert result == pytest.approx(5.0)
    assert type(result) is float

def test_ignores_rows_with_missing_values():
    nan = float("nan")
    df = pd.DataFrame({
        "FANTASY_POINTS": [40, nan, 30, 20],
        "FP_AVG_5": [35, 100, nan, 30],
    })

    result = evaluate_baseline(df)

    # Only the first and last rows are valid: (5 + 10) / 2
    assert result == pytest.approx(7.5)

def test_uses_custom_prediction_column():
    df = pd.DataFrame({
        "FANTASY_POINTS": [40, 30],
        "FP_AVG_5": [float("nan"), float("nan")],
        "FP_AVG_10": [35, 40],
    })

    result = evaluate_baseline(df, prediction_column="FP_AVG_10")

    assert result == pytest.approx(7.5)

@pytest.mark.parametrize(
    "actual, predicted",
    [
        ([], []),
        ([40, float("nan")], [float("nan"), 30]),
    ],
)
def test_rejects_no_valid_rows(actual, predicted):
    df = pd.DataFrame({
        "FANTASY_POINTS": actual,
        "FP_AVG_5": predicted,
    })

    with pytest.raises(ValueError):
        evaluate_baseline(df)

def test_evaluation_preserves_original(evaluation_logs):
    original = evaluation_logs.copy(deep=True)

    evaluate_baseline(evaluation_logs)

    pd.testing.assert_frame_equal(evaluation_logs, original)