import pytest
from fantasy_basketball.nba_data import get_player_id
import pandas as pd
import requests
from unittest.mock import MagicMock
from fantasy_basketball import nba_data

def test_get_player_id():
    player_id = get_player_id("Nikola Jokic")

    assert player_id == 203999
    assert isinstance(player_id, int)

def test_invalid_player_name():
    with pytest.raises(ValueError, match="Player not found"):
        get_player_id("Nonexistent Basketball Player")

def test_player_name_case_insensitive():
    player_id = get_player_id("nikola jokic")

    assert player_id == 203999

def test_get_player_game_logs_success(monkeypatch):
    expected_df = pd.DataFrame({
        "GAME_DATE": ["Apr 12, 2026"],
        "PTS": [23],
        "REB": [8],
        "AST": [1]
    })

    mock_endpoint = MagicMock()
    mock_endpoint.player_game_log.get_data_frame.return_value = expected_df

    mock_constructor = MagicMock(return_value=mock_endpoint)

    monkeypatch.setattr(
        nba_data.playergamelog,
        "PlayerGameLog",
        mock_constructor
    )

    result = nba_data.get_player_game_logs(203999, "2025-26")

    pd.testing.assert_frame_equal(result, expected_df)

def test_get_player_game_logs_parameters(monkeypatch):
    mock_constructor = MagicMock()

    monkeypatch.setattr(
        nba_data.playergamelog,
        "PlayerGameLog",
        mock_constructor
    )

    nba_data.get_player_game_logs(203999, "2025-26")

    mock_constructor.assert_called_once_with(
        player_id=203999,
        season="2025-26",
        timeout=10
    )

def test_get_player_game_logs_timeout(monkeypatch):
    mock_constructor = MagicMock(
        side_effect=requests.exceptions.Timeout("Request timed out")
    )

    monkeypatch.setattr(
        nba_data.playergamelog,
        "PlayerGameLog",
        mock_constructor
    )

    with pytest.raises(requests.exceptions.Timeout):
        nba_data.get_player_game_logs(203999, "2025-26")

def test_get_player_game_logs_connection_error(monkeypatch):
    mock_constructor = MagicMock(
        side_effect=requests.exceptions.ConnectionError("Connection failed")
    )

    monkeypatch.setattr(
        nba_data.playergamelog,
        "PlayerGameLog",
        mock_constructor
    )

    with pytest.raises(requests.exceptions.ConnectionError):
        nba_data.get_player_game_logs(203999, "2025-26")

def test_get_player_game_logs_empty(monkeypatch):
    expected_df = pd.DataFrame()

    mock_endpoint = MagicMock()
    mock_endpoint.player_game_log.get_data_frame.return_value = expected_df

    monkeypatch.setattr(
        nba_data.playergamelog,
        "PlayerGameLog",
        MagicMock(return_value=mock_endpoint)
    )

    result = nba_data.get_player_game_logs(203999, "2025-26")

    assert isinstance(result, pd.DataFrame)
    assert result.empty

@pytest.fixture
def mocked_schedule(monkeypatch):
    endpoint = MagicMock()
    constructor = MagicMock(return_value=endpoint)

    monkeypatch.setattr(
        nba_data.scheduleleaguev2,
        "ScheduleLeagueV2",
        constructor,
    )

    return constructor, endpoint

def test_get_season_schedule_returns_data(mocked_schedule):
    constructor, endpoint = mocked_schedule
    expected = pd.DataFrame({
        "gameId": ["0001"],
        "gameDate": ["11/01/2025 00:00:00"],
        "homeTeam_teamId": [1],
        "awayTeam_teamId": [2],
    })
    endpoint.season_games.get_data_frame.return_value = expected

    result = nba_data.get_season_schedule("2025-26")

    pd.testing.assert_frame_equal(result, expected)
    constructor.assert_called_once_with(
        league_id="00",
        season="2025-26",
        timeout=10,
    )

def test_get_season_schedule_handles_empty_results(mocked_schedule):
    _, endpoint = mocked_schedule
    expected = pd.DataFrame()
    endpoint.season_games.get_data_frame.return_value = expected

    result = nba_data.get_season_schedule("2025-26")

    pd.testing.assert_frame_equal(result, expected)

@pytest.mark.parametrize(
    "error_type",
    [
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
    ],
)
def test_get_season_schedule_propagates_errors(mocked_schedule, error_type):
    constructor, _ = mocked_schedule
    constructor.side_effect = error_type("Request failed")

    with pytest.raises(error_type, match="Request failed"):
        nba_data.get_season_schedule("2025-26")
