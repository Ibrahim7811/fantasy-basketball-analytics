from unittest.mock import Mock
import pytest
import pandas as pd
from fantasy_basketball import ingestion
from fantasy_basketball.database import get_connection, initialize_database

TEAM_QUERY = """
    SELECT team_id, name, abbreviation
    FROM teams
    ORDER BY team_id
"""
@pytest.fixture
def connection(tmp_path):
    conn = get_connection(str(tmp_path / "test.db"))
    initialize_database(conn)

    try:
        yield conn
    finally:
        conn.close()

@pytest.fixture
def mocked_teams(monkeypatch):
    getter = Mock(return_value=[
        {"id": 1, "full_name": "Home Team", "abbreviation": "HOM"},
        {"id": 2, "full_name": "Away Team", "abbreviation": "AWY"},
    ])

    monkeypatch.setattr(ingestion.nba_teams, "get_teams", getter)

    return getter

def test_sync_teams_saves_records_and_returns_lookup(connection, mocked_teams):
    result = ingestion.sync_teams(connection)

    assert result == {"HOM": 1, "AWY": 2}
    assert connection.execute(TEAM_QUERY).fetchall() == [
        (1, "Home Team", "HOM"),
        (2, "Away Team", "AWY"),
    ]

def test_sync_teams_updates_without_duplicates(connection, mocked_teams):
    ingestion.sync_teams(connection)

    mocked_teams.return_value[0]["full_name"] = "Updated Home Team"
    mocked_teams.return_value[0]["abbreviation"] = "UPD"

    result = ingestion.sync_teams(connection)

    assert result == {"UPD": 1, "AWY": 2}
    assert connection.execute(TEAM_QUERY).fetchall() == [
        (1, "Updated Home Team", "UPD"),
        (2, "Away Team", "AWY"),
    ]

@pytest.mark.parametrize("existing_records", [False, True])
def test_sync_teams_leaves_commit_to_caller(connection, mocked_teams, existing_records):
    if existing_records:
        ingestion.sync_teams(connection)
        connection.commit()

    original = connection.execute(TEAM_QUERY).fetchall()

    mocked_teams.return_value[0]["full_name"] = "Changed Name"

    ingestion.sync_teams(connection)
    connection.rollback()

    assert connection.execute(TEAM_QUERY).fetchall() == original

@pytest.fixture
def team_lookup():
    return {"DEN": 1, "LAL": 2}

@pytest.mark.parametrize(
    "matchup, expected",
    [
        ("DEN vs. LAL", (1, 1, 2)),
        ("DEN @ LAL", (1, 2, 1)),
        ("LAL vs. DEN", (2, 2, 1)),
        ("LAL @ DEN", (2, 1, 2)),
        ("  DEN\t @  LAL  ", (1, 2, 1)),
    ],
)

def test_parse_matchup_valid_formats(team_lookup, matchup, expected):
    result = ingestion.parse_matchup(matchup, team_lookup)
    assert result == expected

@pytest.mark.parametrize(
    "matchup, error",
    [
        ("", "Invalid matchup format"),
        ("DEN", "Invalid matchup format"),
        ("DEN LAL", "Invalid matchup format"),
        ("DEN vs. LAL EXTRA", "Invalid matchup format"),
        ("DEN vs LAL", "Invalid matchup separator"),
        ("DEN - LAL", "Invalid matchup separator"),
        ("XXX vs. LAL", "Unknown team abbreviation"),
        ("DEN @ XXX", "Unknown team abbreviation"),
        ("DEN vs. DEN", "A team cannot play itself"),
        ("LAL @ LAL", "A team cannot play itself"),
    ],
)

def test_parse_matchup_rejects_invalid_input(team_lookup, matchup, error):
    with pytest.raises(ValueError, match=error):
        ingestion.parse_matchup(matchup, team_lookup)

def test_parse_matchup_rejects_identical_team_ids():
    # Different abbreviations still cannot represent the same team.
    lookup = {"DEN": 1, "ALIAS": 1}

    with pytest.raises(ValueError, match="A team cannot play itself"):
        ingestion.parse_matchup("DEN vs. ALIAS", lookup)

SCHEDULE_QUERY = """
    SELECT game_id, date, season, home_team, away_team
    FROM games
    ORDER BY game_id
"""
@pytest.fixture
def saved_team_lookup(connection, mocked_teams):
    lookup = ingestion.sync_teams(connection)
    connection.commit()
    return lookup

@pytest.fixture
def mocked_schedule_feed(monkeypatch):
    raw = pd.DataFrame(
        [
            ("0022500002", "11/05/2025 00:00:00", 2, 1),
            ("0022500001", "11/01/2025 00:00:00", 1, 2),
            ("0012500001", "10/02/2025 00:00:00", 1, 2),
            ("0022500003", "11/07/2025 00:00:00", 999, 2),
        ],
        columns=[
            "gameId",
            "gameDate",
            "homeTeam_teamId",
            "awayTeam_teamId",
        ],
    )
    getter = Mock(return_value=raw)
    monkeypatch.setattr(ingestion, "get_season_schedule", getter)
    return getter

def test_sync_schedule_saves_valid_games(connection, saved_team_lookup, mocked_schedule_feed):
    count = ingestion.sync_schedule(
        connection, "2025-26", saved_team_lookup
    )

    assert count == 2
    assert connection.execute(SCHEDULE_QUERY).fetchall() == [
        ("0022500001", "2025-11-01", "2025-26", 1, 2),
        ("0022500002", "2025-11-05", "2025-26", 2, 1),
    ]
    mocked_schedule_feed.assert_called_once_with("2025-26")

def test_sync_schedule_updates_without_duplicates(connection, saved_team_lookup, mocked_schedule_feed):
    ingestion.sync_schedule(connection, "2025-26", saved_team_lookup)

    raw = mocked_schedule_feed.return_value
    raw.loc[0, "gameDate"] = "11/06/2025 00:00:00"
    raw.loc[0, "homeTeam_teamId"] = 1
    raw.loc[0, "awayTeam_teamId"] = 2

    count = ingestion.sync_schedule(
        connection, "2025-26", saved_team_lookup
    )

    # The count includes updates, not just new records.
    assert count == 2
    assert connection.execute(SCHEDULE_QUERY).fetchall() == [
        ("0022500001", "2025-11-01", "2025-26", 1, 2),
        ("0022500002", "2025-11-06", "2025-26", 1, 2),
    ]

@pytest.mark.parametrize("empty_source", [True, False])
def test_sync_schedule_handles_no_matching_games(connection, saved_team_lookup, mocked_schedule_feed, empty_source):
    raw = mocked_schedule_feed.return_value
    mocked_schedule_feed.return_value = (
        pd.DataFrame() if empty_source else raw.iloc[[2]].copy()
    )

    count = ingestion.sync_schedule(
        connection, "2025-26", saved_team_lookup
    )

    assert count == 0
    assert connection.execute(SCHEDULE_QUERY).fetchall() == []

@pytest.mark.parametrize("existing_records", [False, True])
def test_sync_schedule_leaves_commit_to_caller(connection, saved_team_lookup, mocked_schedule_feed, existing_records):
    if existing_records:
        ingestion.sync_schedule(
            connection, "2025-26", saved_team_lookup
        )
        connection.commit()

    original = connection.execute(SCHEDULE_QUERY).fetchall()
    mocked_schedule_feed.return_value.loc[
        0, "gameDate"
    ] = "11/06/2025 00:00:00"

    ingestion.sync_schedule(connection, "2025-26", saved_team_lookup)
    connection.rollback()

    assert connection.execute(SCHEDULE_QUERY).fetchall() == original

PLAYER_STATS_QUERY = """
    SELECT game_id, player_id, team_id, minutes,
           pts, reb, ast, stl, blk, tov
    FROM player_game_stats
    ORDER BY game_id, player_id
"""


@pytest.fixture
def player_log_source(connection, saved_team_lookup, mocked_schedule_feed, monkeypatch):
    lookup = dict(saved_team_lookup)
    lookup["OTH"] = 3

    connection.execute("""
        INSERT INTO teams (team_id, name, abbreviation)
        VALUES (3, 'Other Team', 'OTH')
    """)
    ingestion.sync_schedule(connection, "2025-26", lookup)
    connection.commit()

    logs = pd.DataFrame({
        "Game_ID": ["0022500002", "0022500001"],
        "MATCHUP": ["HOM @ AWY", "HOM vs. AWY"],
        "MIN": [36.5, 32.5],
        "PTS": [30, 20],
        "REB": [9, 8],
        "AST": [6, 5],
        "STL": [2, 1],
        "BLK": [3, 2],
        "TOV": [4, 3],
    })

    player_lookup = Mock(return_value=10)
    log_fetcher = Mock(return_value=logs)

    monkeypatch.setattr(ingestion, "get_player_id", player_lookup)
    monkeypatch.setattr(ingestion, "get_player_game_logs", log_fetcher)

    return lookup, player_lookup, log_fetcher

@pytest.mark.parametrize("first_matchup", ["HOM @ AWY", "HOM vs. AWY"])
def test_sync_player_logs_saves_records(connection, player_log_source, first_matchup):
    lookup, player_lookup, log_fetcher = player_log_source
    log_fetcher.return_value.loc[0, "MATCHUP"] = first_matchup

    count = ingestion.sync_player_game_logs(
        connection, "Test Player", "2025-26", lookup
    )

    assert count == 2
    assert connection.execute(PLAYER_STATS_QUERY).fetchall() == [
        ("0022500001", 10, 1, 32.5, 20, 8, 5, 1, 2, 3),
        ("0022500002", 10, 1, 36.5, 30, 9, 6, 2, 3, 4),
    ]
    assert connection.execute(
        "SELECT player_id, name FROM players"
    ).fetchall() == [(10, "Test Player")]

    # The saved schedule remains authoritative for home and away.
    assert connection.execute("""
        SELECT home_team, away_team
        FROM games WHERE game_id = '0022500002'
    """).fetchone() == (2, 1)

    player_lookup.assert_called_once_with("Test Player")
    log_fetcher.assert_called_once_with(10, "2025-26")

def test_sync_player_logs_updates_without_duplicates(connection, player_log_source):
    lookup, _, log_fetcher = player_log_source
    ingestion.sync_player_game_logs(
        connection, "Test Player", "2025-26", lookup
    )

    log_fetcher.return_value.loc[0, "PTS"] = 45
    log_fetcher.return_value.loc[0, "MATCHUP"] = "AWY vs. HOM"

    count = ingestion.sync_player_game_logs(
        connection, "Test Player", "2025-26", lookup
    )

    assert count == 2
    assert connection.execute("""
        SELECT game_id, team_id, pts
        FROM player_game_stats
        ORDER BY game_id
    """).fetchall() == [
        ("0022500001", 1, 20),
        ("0022500002", 2, 45),
    ]

@pytest.mark.parametrize(
    "row_index, column, value, error",
    [
        (0, "Game_ID", "missing", "missing from the schedule"),
        (0, "MATCHUP", "OTH @ AWY", "Teams do not match"),
        (1, "Game_ID", "missing", "missing from the schedule"),
    ],
)
def test_sync_player_logs_failure_rolls_back_batch(connection, player_log_source, row_index, column, value, error):
    lookup, _, log_fetcher = player_log_source
    log_fetcher.return_value.loc[row_index, column] = value

    with pytest.raises(ValueError, match=error):
        with connection:
            ingestion.sync_player_game_logs(connection, "Test Player", "2025-26", lookup)

    assert connection.execute(PLAYER_STATS_QUERY).fetchall() == []
    assert connection.execute(
        "SELECT COUNT(*) FROM players"
    ).fetchone()[0] == 0


def test_sync_player_logs_handles_empty_results(connection, player_log_source):
    lookup, _, log_fetcher = player_log_source
    log_fetcher.return_value = pd.DataFrame()

    count = ingestion.sync_player_game_logs(
        connection, "Test Player", "2025-26", lookup
    )

    assert count == 0
    assert connection.execute(PLAYER_STATS_QUERY).fetchall() == []
    assert connection.execute(
        "SELECT player_id, name FROM players"
    ).fetchall() == [(10, "Test Player")]

def test_sync_player_logs_leaves_commit_to_caller(connection, player_log_source):
    lookup, _, _ = player_log_source

    ingestion.sync_player_game_logs(connection, "Test Player", "2025-26", lookup)
    connection.rollback()

    assert connection.execute(PLAYER_STATS_QUERY).fetchall() == []
    assert connection.execute(
        "SELECT COUNT(*) FROM players"
    ).fetchone()[0] == 0