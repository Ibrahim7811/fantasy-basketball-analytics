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