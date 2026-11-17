import sqlite3
import pytest
from fantasy_basketball.database import get_connection, initialize_database, upsert_team


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