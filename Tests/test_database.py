import sqlite3
import pytest
import pandas as pd
from fantasy_basketball.database import get_connection, initialize_database, upsert_team, upsert_player, upsert_game, upsert_player_game_stats, load_player_game_logs
from fantasy_basketball.data_processing import prepare_game_logs

@pytest.fixture
def database(tmp_path):
    db_path = tmp_path / "test.db"
    connection = get_connection(str(db_path))

    try:
        yield connection, db_path
    finally:
        connection.close()

def test_creates_database_file(database):
    _, db_path = database

    assert db_path.is_file()

def test_returns_usable_connection(database):
    connection, _ = database

    assert isinstance(connection, sqlite3.Connection)

    result = connection.execute("SELECT ? + ?", (2, 3)).fetchone()

    assert result == (5,)

def test_enforces_foreign_keys(database):
    connection, _ = database

    connection.execute("""
        CREATE TABLE parent (
            id INTEGER PRIMARY KEY
        )
    """)
    connection.execute("""
        CREATE TABLE child (
            parent_id INTEGER REFERENCES parent(id)
        )
    """)

    # A reference to an existing parent succeeds.
    connection.execute("INSERT INTO parent (id) VALUES (1)")
    connection.execute("INSERT INTO child (parent_id) VALUES (1)")

    # A reference to a nonexistent parent must fail.
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO child (parent_id) VALUES (999)"
        )

STATS_INSERT = """
    INSERT INTO player_game_stats (
        game_id, player_id, team_id,
        minutes, pts, reb, ast, stl, blk, tov
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

@pytest.fixture
def initialized_connection(database):
    connection, _ = database
    initialize_database(connection)

    connection.executescript("""
        INSERT INTO teams (team_id, name, abbreviation)
        VALUES
            (1, 'Home Team', 'HOM'),
            (2, 'Away Team', 'AWY');

        INSERT INTO players (player_id, name)
        VALUES (10, 'Test Player');

        INSERT INTO games (
            game_id, date, season, home_team, away_team
        )
        VALUES ('0001', '2026-11-01', '2026-27', 1, 2);
    """)

    return connection

def test_initialization_creates_tables(database):
    connection, _ = database
    initialize_database(connection)

    rows = connection.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
    """).fetchall()

    assert {row[0] for row in rows} == {
        "teams", "players", "games", "player_game_stats",
    }

def test_reinitialization_preserves_data(initialized_connection):
    connection = initialized_connection

    initialize_database(connection)

    rows = connection.execute(
        "SELECT game_id FROM games"
    ).fetchall()

    # Existing data and leading zeros must survive.
    assert rows == [("0001",)]

def test_rejects_duplicate_player_game(initialized_connection):
    connection = initialized_connection
    stats = ("0001", 10, 1, 32.5, 20, 8, 5, 1, 2, 3)

    connection.execute(STATS_INSERT, stats)

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(STATS_INSERT, stats)

@pytest.mark.parametrize(
    "game_id, player_id, team_id",
    [
        ("missing", 10, 1),
        ("0001", 999, 1),
        ("0001", 10, 999),
    ],
)
def test_stats_require_existing_references(
    initialized_connection, game_id, player_id, team_id
):
    connection = initialized_connection
    stats = (
        game_id, player_id, team_id,
        32.5, 20, 8, 5, 1, 2, 3,
    )

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(STATS_INSERT, stats)

def test_upsert_team_inserts_new_team(initialized_connection):
    connection = initialized_connection

    upsert_team(connection, 3, "New Team", "NEW")

    row = connection.execute("""
        SELECT team_id, name, abbreviation
        FROM teams
        WHERE team_id = ?
    """, (3,)).fetchone()

    assert row == (3, "New Team", "NEW")

def test_upsert_team_updates_without_duplicates(initialized_connection):
    connection = initialized_connection

    upsert_team(connection, 1, "Updated Home Team", "UPD")

    row = connection.execute("""
        SELECT name, abbreviation
        FROM teams
        WHERE team_id = ?
    """, (1,)).fetchone()

    assert row == ("Updated Home Team", "UPD")
    assert connection.execute(
        "SELECT COUNT(*) FROM teams"
    ).fetchone()[0] == 2

@pytest.mark.parametrize("team_id", [2, 3])
def test_upsert_team_rejects_conflicting_abbreviation(
    initialized_connection, team_id
):
    connection = initialized_connection

    # HOM already belongs to team 1.
    # Check both updating team 2 and inserting team 3.
    with pytest.raises(sqlite3.IntegrityError):
        upsert_team(connection, team_id, "Conflicting Team", "HOM")

def test_upsert_team_leaves_commit_to_caller(initialized_connection):
    connection = initialized_connection

    upsert_team(connection, 3, "New Team", "NEW")
    connection.rollback()

    row = connection.execute(
        "SELECT team_id FROM teams WHERE team_id = ?",
        (3,),
    ).fetchone()

    assert row is None

def test_upsert_player_inserts_new_player(initialized_connection):
    connection = initialized_connection

    upsert_player(connection, 20, "New Player")

    row = connection.execute("""
        SELECT player_id, name
        FROM players
        WHERE player_id = ?
    """, (20,)).fetchone()

    assert row == (20, "New Player")

def test_upsert_player_updates_without_duplicates(initialized_connection):
    connection = initialized_connection

    # Player 10 already exists in the fixture.
    upsert_player(connection, 10, "Updated Player")

    rows = connection.execute("""
        SELECT player_id, name
        FROM players
        ORDER BY player_id
    """).fetchall()

    assert rows == [(10, "Updated Player")]

def test_upsert_player_leaves_commit_to_caller(initialized_connection):
    connection = initialized_connection

    upsert_player(connection, 20, "New Player")
    connection.rollback()

    row = connection.execute(
        "SELECT player_id FROM players WHERE player_id = ?",
        (20,),
    ).fetchone()

    assert row is None

def test_upsert_game_inserts_and_preserves_id(initialized_connection):
    connection = initialized_connection

    upsert_game(connection, "0002", "2026-11-02", "2026-27", 1, 2)

    row = connection.execute("""
        SELECT game_id, date, season, home_team, away_team
        FROM games
        WHERE game_id = ?
    """, ("0002",)).fetchone()

    assert row == ("0002", "2026-11-02", "2026-27", 1, 2)

def test_upsert_game_updates_without_duplicates(initialized_connection):
    connection = initialized_connection

    # Update every non-key field of the existing game.
    upsert_game(connection, "0001", "2026-03-01", "2025-26", 2, 1)

    rows = connection.execute("""
        SELECT game_id, date, season, home_team, away_team
        FROM games
    """).fetchall()

    assert rows == [("0001", "2026-03-01", "2025-26", 2, 1)]

@pytest.mark.parametrize("game_id", ["0001", "0002"])
@pytest.mark.parametrize(
    "home_team, away_team",
    [
        (999, 2),  # Missing home team
        (1, 999),  # Missing away team
        (1, 1),    # A team cannot play itself
    ],
)

def test_upsert_game_rejects_invalid_teams(initialized_connection, game_id, home_team, away_team):
    connection = initialized_connection

    # Exercise both updating an existing game and inserting a new one.
    with pytest.raises(sqlite3.IntegrityError):
        upsert_game(
            connection, game_id, "2026-11-02", "2026-27",
            home_team, away_team,
        )

@pytest.mark.parametrize("game_id", ["0001", "0002"])
def test_upsert_game_leaves_commit_to_caller(initialized_connection, game_id):
    connection = initialized_connection
    query = """
        SELECT game_id, date, season, home_team, away_team
        FROM games
        ORDER BY game_id
    """
    original = connection.execute(query).fetchall()

    upsert_game(connection, game_id, "2026-11-02", "2026-27", 2, 1)
    connection.rollback()

    assert connection.execute(query).fetchall() == original

STATS_SELECT = """
    SELECT game_id, player_id, team_id, minutes,
           pts, reb, ast, stl, blk, tov
    FROM player_game_stats
    ORDER BY game_id, player_id
"""

SAMPLE_GAME_STATS = ("0001", 10, 1, 32.5, 20, 8, 5, 1, 2, 3)
UPDATED_GAME_STATS = ("0001", 10, 2, 35.5, 30, 9, 6, 2, 3, 4)

def test_upsert_stats_inserts_record(initialized_connection):
    connection = initialized_connection

    upsert_player_game_stats(connection, *SAMPLE_GAME_STATS)

    rows = connection.execute(STATS_SELECT).fetchall()

    assert rows == [SAMPLE_GAME_STATS]

def test_upsert_stats_updates_without_duplicates(initialized_connection):
    connection = initialized_connection

    upsert_player_game_stats(connection, *SAMPLE_GAME_STATS)
    upsert_player_game_stats(connection, *UPDATED_GAME_STATS)

    rows = connection.execute(STATS_SELECT).fetchall()

    # Every non-key field changes, but only one row remains.
    assert rows == [UPDATED_GAME_STATS]

def test_upsert_stats_keeps_player_game_pairs_separate(initialized_connection):
    connection = initialized_connection
    connection.executescript("""
        INSERT INTO players (player_id, name)
        VALUES (20, 'Second Player');

        INSERT INTO games (
            game_id, date, season, home_team, away_team
        )
        VALUES ('0002', '2026-11-02', '2026-27', 1, 2);
    """)

    for game_id, player_id, points in [
        ("0001", 10, 20),
        ("0001", 20, 30),
        ("0002", 10, 40),
    ]:
        upsert_player_game_stats(
            connection, game_id, player_id, 1,
            32.5, points, 8, 5, 1, 2, 3,
        )

    rows = connection.execute("""
        SELECT game_id, player_id, pts
        FROM player_game_stats
        ORDER BY game_id, player_id
    """).fetchall()

    assert rows == [
        ("0001", 10, 20),
        ("0001", 20, 30),
        ("0002", 10, 40),
    ]

@pytest.mark.parametrize(
    "game_id, player_id, team_id",
    [
        ("missing", 10, 1),
        ("0001", 999, 1),
        ("0001", 10, 999),
    ],
)
def test_upsert_stats_rejects_missing_references(initialized_connection, game_id, player_id, team_id):
    connection = initialized_connection

    with pytest.raises(sqlite3.IntegrityError):
        upsert_player_game_stats(
            connection, game_id, player_id, team_id,
            32.5, 20, 8, 5, 1, 2, 3,
        )

@pytest.mark.parametrize("existing_record", [False, True])
def test_upsert_stats_leaves_commit_to_caller(initialized_connection, existing_record):
    connection = initialized_connection

    if existing_record:
        upsert_player_game_stats(connection, *SAMPLE_GAME_STATS)
        connection.commit()

    original = connection.execute(STATS_SELECT).fetchall()

    upsert_player_game_stats(connection, *UPDATED_GAME_STATS)
    connection.rollback()

    assert connection.execute(STATS_SELECT).fetchall() == original

LOG_COLUMNS = [
    "GAME_ID", "PLAYER_ID", "TEAM_ID", "GAME_DATE",
    "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
]

@pytest.fixture
def stored_game_logs(initialized_connection):
    connection = initialized_connection

    connection.executescript("""
        INSERT INTO players (player_id, name)
        VALUES (20, 'Second Player');

        INSERT INTO games (
            game_id, date, season, home_team, away_team
        )
        VALUES
            ('0002', '2026-11-05', '2026-27', 1, 2),
            ('0003', '2026-11-03', '2026-27', 1, 2),
            ('old0001', '2025-11-01', '2025-26', 1, 2);
    """)

    # Deliberately unsorted; includes another player and season.
    connection.executemany(STATS_INSERT, [
        ("0002", 10, 2, 36.5, 30, 9, 6, 2, 3, 4),
        ("0001", 10, 1, 32.5, 20, 8, 5, 1, 2, 3),
        ("0003", 10, 1, 34.0, 25, 7, 4, 0, 1, 2),
        ("0001", 20, 2, 28.0, 99, 1, 2, 3, 4, 5),
        ("old0001", 10, 1, 30.0, 77, 5, 6, 1, 0, 2),
    ])

    return connection

def test_load_logs_filters_sorts_and_preserves_values(stored_game_logs):
    connection = stored_game_logs

    result = load_player_game_logs(connection, 10, "2026-27")

    expected = pd.DataFrame({
        "GAME_ID": ["0001", "0003", "0002"],
        "PLAYER_ID": [10, 10, 10],
        "TEAM_ID": [1, 1, 2],
        "GAME_DATE": pd.to_datetime([
            "2026-11-01",
            "2026-11-03",
            "2026-11-05",
        ]),
        "MIN": [32.5, 34.0, 36.5],
        "PTS": [20, 25, 30],
        "REB": [8, 7, 9],
        "AST": [5, 4, 6],
        "STL": [1, 0, 2],
        "BLK": [2, 1, 3],
        "TOV": [3, 2, 4],
    })

    pd.testing.assert_frame_equal(result, expected)

    # Loading must leave the connection open.
    assert connection.execute("SELECT 1").fetchone() == (1,)

@pytest.mark.parametrize(
    "player_id, season",
    [
        (999, "2026-27"),
        (10, "1999-00"),
    ],
)
def test_load_logs_returns_empty_dataframe(stored_game_logs, player_id, season):
    result = load_player_game_logs(
        stored_game_logs, player_id, season
    )

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert result.columns.tolist() == LOG_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(result["GAME_DATE"])

def test_loaded_logs_work_with_processing_pipeline(stored_game_logs):
    logs = load_player_game_logs(stored_game_logs, 10, "2026-27")

    result = prepare_game_logs(
        logs,
        scoring_weights={"PTS": 1.0, "TOV": -2.0},
        windows=(2,),
    )

    assert result["FANTASY_POINTS"].tolist() == pytest.approx(
        [14.0, 21.0, 22.0]
    )

    assert pd.isna(result["FP_AVG_2"].iloc[0])
    assert result["FP_AVG_2"].iloc[1:].tolist() == pytest.approx(
        [14.0, 17.5]
    )

    assert pd.isna(result["REST_DAYS"].iloc[0])
    assert result["REST_DAYS"].iloc[1:].tolist() == [1.0, 1.0]