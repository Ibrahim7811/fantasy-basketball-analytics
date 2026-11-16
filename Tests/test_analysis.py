import pandas as pd
import pytest
from unittest.mock import Mock
from fantasy_basketball import analysis

@pytest.fixture
def mocked_nba(monkeypatch):
    raw_logs = pd.DataFrame({
        "GAME_DATE": [
            "Nov 06, 2026",
            "Nov 01, 2026",
            "Nov 05, 2026",
            "Nov 02, 2026",
            "Nov 04, 2026",
            "Nov 03, 2026",
        ],
        "PTS": [60, 10, 50, 20, 40, 30],
        "TOV": [1, 1, 1, 1, 1, 1],
    })

    lookup = Mock(return_value=203999)
    fetch = Mock(return_value=raw_logs)

    monkeypatch.setattr(analysis, "get_player_id", lookup)
    monkeypatch.setattr(analysis, "get_player_game_logs", fetch)

    return raw_logs, lookup, fetch

def test_analyze_player_complete_pipeline(mocked_nba):
    raw_logs, lookup, fetch = mocked_nba
    original = raw_logs.copy(deep=True)

    result = analysis.analyze_player(
        "Nikola Jokic",
        "2025-26",
        {"PTS": 1.0, "TOV": -2.0},
    )

    # Sorted fantasy scores: 8, 18, 28, 38, 48, 58
    # Next prediction: average of the last five = 38
    assert result["games_analyzed"] == 6
    assert result["projected_fp"] == pytest.approx(38.0)

    # Historical absolute errors: 10, 15, 20, 25, 30
    assert result["baseline_mae"] == pytest.approx(20.0)

    lookup.assert_called_once_with("Nikola Jokic")
    fetch.assert_called_once_with(203999, "2025-26")
    pd.testing.assert_frame_equal(raw_logs, original)

@pytest.mark.parametrize("game_count", [0, 1])
def test_analyze_player_rejects_insufficient_games(mocked_nba, game_count):
    _, _, fetch = mocked_nba
    fetch.return_value = pd.DataFrame({
        "PTS": list(range(game_count)),
    })

    with pytest.raises(ValueError, match="At least two games"):
        analysis.analyze_player(
            "Nikola Jokic",
            "2025-26",
            {"PTS": 1.0},
        )