from datetime import date
import psycopg
import pytest
from psycopg import sql
from fantasy_basketball.database import get_connection, initialize_database
from fantasy_basketball.migration import migrate_sqlite_to_postgres

@pytest.fixture
def sqlite_source():
    connection = get_connection(":memory:")
    initialize_database(connection)
    connection.executemany("INSERT INTO teams VALUES (?, ?, ?)", [
        (1, "Home Team", "HOM"),
        (2, "Away Team", "AWY")])
    connection.executemany("INSERT INTO players VALUES (?, ?)", [
        (10, "First Player"),
        (20, "Second Player")])
    connection.executemany("INSERT INTO games VALUES (?, ?, ?, ?, ?)", [
        ("0022500001", "2026-02-03", "2025-26", 1, 2),
        ("0022500002", "2026-02-05", "2025-26", 2, 1)])
    connection.executemany("INSERT INTO player_game_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
        ("0022500001", 10, 1, 32.123456789, 20, 8, 5, 1, 2, 3),
        ("0022500001", 20, 2, 30.5, 18, 6, 4, 2, 1, 2),
        ("0022500002", 10, 1, 35.25, 30, 9, 6, 2, 3, 4)])
    connection.commit()
    try:
        yield connection
    finally:
        connection.close()

def test_migration_copies_all_records_and_preserves_source(sqlite_source, postgres_connection):
    tables = ("teams", "players", "games", "player_game_stats")
    original = {table: sqlite_source.execute(f"SELECT * FROM {table} ORDER BY 1, 2").fetchall() for table in tables}
    counts = migrate_sqlite_to_postgres(sqlite_source, postgres_connection)
    assert counts == {"teams": 2, "players": 2, "games": 2, "player_game_stats": 3}
    for table in tables:
        query = sql.SQL("SELECT * FROM {} ORDER BY 1, 2").format(sql.Identifier(table))
        actual = postgres_connection.execute(query).fetchall()
        expected = original[table]
        if table == "games":
            expected = [(row[0], date.fromisoformat(row[1]), *row[2:]) for row in expected]
        assert actual == expected
        assert sqlite_source.execute(f"SELECT * FROM {table} ORDER BY 1, 2").fetchall() == original[table]

def test_migration_rejects_repeat_without_changing_destination(sqlite_source, postgres_connection):
    expected_counts = migrate_sqlite_to_postgres(sqlite_source, postgres_connection)
    with pytest.raises(ValueError, match="Destination table teams must be empty"):
        migrate_sqlite_to_postgres(sqlite_source, postgres_connection)
    for table, expected_count in expected_counts.items():
        query = sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table))
        assert postgres_connection.execute(query).fetchone() == (expected_count,)
    row = postgres_connection.execute(
        "SELECT minutes, pts FROM player_game_stats WHERE game_id = %s AND player_id = %s",("0022500001", 10)).fetchone()
    assert row == (32.123456789, 20)

def test_migration_preserves_preexisting_destination_data(sqlite_source, postgres_connection):
    with postgres_connection.transaction():
        postgres_connection.execute("INSERT INTO teams VALUES (%s, %s, %s)", (99, "Existing Team", "OLD"))
    with pytest.raises(ValueError, match="Destination table teams must be empty"):
        migrate_sqlite_to_postgres(sqlite_source, postgres_connection)
    assert postgres_connection.execute("SELECT * FROM teams").fetchall() == [(99, "Existing Team", "OLD")]
    assert postgres_connection.execute("SELECT COUNT(*) FROM players").fetchone() == (0,)
    assert postgres_connection.execute("SELECT COUNT(*) FROM games").fetchone() == (0,)
    assert postgres_connection.execute("SELECT COUNT(*) FROM player_game_stats").fetchone() == (0,)

def test_migration_handles_empty_source(sqlite_source, postgres_connection):
    for table in ("player_game_stats", "games", "players", "teams"):
        sqlite_source.execute(f"DELETE FROM {table}")
    sqlite_source.commit()
    counts = migrate_sqlite_to_postgres(sqlite_source, postgres_connection)
    assert counts == {"teams": 0, "players": 0, "games": 0, "player_game_stats": 0}
    for table in counts:
        query = sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table))
        assert postgres_connection.execute(query).fetchone() == (0,)

@pytest.mark.parametrize("problem, expected_error", [
    ("foreign_key", psycopg.errors.ForeignKeyViolation),
    ("invalid_date", ValueError)
])
def test_migration_rolls_back_entire_copy_on_failure(sqlite_source, postgres_connection, problem, expected_error):
    if problem == "foreign_key":
        sqlite_source.execute("PRAGMA foreign_keys = OFF")
        sqlite_source.execute("UPDATE player_game_stats SET team_id = 999")
    else:
        sqlite_source.execute("UPDATE games SET date = 'invalid-date'")
    sqlite_source.commit()
    with pytest.raises(expected_error):
        migrate_sqlite_to_postgres(sqlite_source, postgres_connection)
    for table in ("teams", "players", "games", "player_game_stats"):
        query = sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table))
        assert postgres_connection.execute(query).fetchone() == (0,)