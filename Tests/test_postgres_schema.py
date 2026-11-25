import os
from datetime import date
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4
import psycopg
import pytest
from dotenv import dotenv_values
from psycopg import sql
from fantasy_basketball.postgres_database import get_postgres_connection, initialize_postgres_database

@pytest.fixture
def seeded_postgres(postgres_connection):
    connection = postgres_connection
    with connection.transaction():
        connection.execute("INSERT INTO teams (team_id, name, abbreviation) VALUES (%s, %s, %s)", (1, "Home Team", "HOM"))
        connection.execute("INSERT INTO teams (team_id, name, abbreviation) VALUES (%s, %s, %s)", (2, "Away Team", "AWY"))
        connection.execute("INSERT INTO players (player_id, name) VALUES (%s, %s)", (10, "Test Player"))
        connection.execute(
            "INSERT INTO games (game_id, date, season, home_team, away_team) VALUES (%s, %s, %s, %s, %s)",
            ("0022500001", date(2026, 2, 3), "2025-26", 1, 2))
    return connection

def test_postgres_schema_creates_all_tables(postgres_connection):
    rows = postgres_connection.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema() AND table_type = 'BASE TABLE'"
    ).fetchall()
    assert {row[0] for row in rows} == {"teams", "players", "games", "player_game_stats"}

def test_postgres_schema_initialization_is_repeatable_and_preserves_data(seeded_postgres):
    connection = seeded_postgres
    with connection.transaction():
        initialize_postgres_database(connection)
        initialize_postgres_database(connection)
    assert connection.execute("SELECT COUNT(*) FROM teams").fetchone() == (2,)
    assert connection.execute("SELECT COUNT(*) FROM players").fetchone() == (1,)
    assert connection.execute("SELECT COUNT(*) FROM games").fetchone() == (1,)

def test_postgres_schema_preserves_game_id_and_native_date(seeded_postgres):
    row = seeded_postgres.execute("SELECT game_id, date FROM games").fetchone()
    assert row == ("0022500001", date(2026, 2, 3))

def test_postgres_schema_preserves_minutes_precision(seeded_postgres):
    minutes = 32.123456789
    with seeded_postgres.transaction():
        seeded_postgres.execute(
            "INSERT INTO player_game_stats (game_id, player_id, team_id, minutes, pts, reb, ast, stl, blk, tov) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            ("0022500001", 10, 1, minutes, 20, 8, 5, 1, 2, 3))
    row = seeded_postgres.execute("SELECT minutes FROM player_game_stats").fetchone()
    assert row[0] == minutes
    data_type = seeded_postgres.execute(
        "SELECT data_type FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'player_game_stats' AND column_name = 'minutes'"
    ).fetchone()
    assert data_type == ("double precision",)

def test_postgres_schema_rejects_duplicate_team_abbreviation(seeded_postgres):
    with pytest.raises(psycopg.errors.UniqueViolation):
        with seeded_postgres.transaction():
            seeded_postgres.execute("INSERT INTO teams (team_id, name, abbreviation) VALUES (%s, %s, %s)", (3, "Duplicate Team", "HOM"))

def test_postgres_schema_rejects_game_against_same_team(seeded_postgres):
    with pytest.raises(psycopg.errors.CheckViolation):
        with seeded_postgres.transaction():
            seeded_postgres.execute(
                "INSERT INTO games (game_id, date, season, home_team, away_team) VALUES (%s, %s, %s, %s, %s)",
                ("0022500002", date(2026, 2, 4), "2025-26", 1, 1))

def test_postgres_schema_rejects_unknown_game_team(seeded_postgres):
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with seeded_postgres.transaction():
            seeded_postgres.execute(
                "INSERT INTO games (game_id, date, season, home_team, away_team) VALUES (%s, %s, %s, %s, %s)",
                ("0022500002", date(2026, 2, 4), "2025-26", 1, 999))

@pytest.mark.parametrize("game_id, player_id, team_id", [
    ("missing", 10, 1),
    ("0022500001", 999, 1),
    ("0022500001", 10, 999)
])
def test_postgres_schema_enforces_stats_foreign_keys(seeded_postgres, game_id, player_id, team_id):
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with seeded_postgres.transaction():
            seeded_postgres.execute(
                "INSERT INTO player_game_stats (game_id, player_id, team_id, minutes, pts, reb, ast, stl, blk, tov) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (game_id, player_id, team_id, 32.5, 20, 8, 5, 1, 2, 3))

def test_postgres_schema_enforces_composite_stats_key(seeded_postgres):
    statement = """
        INSERT INTO player_game_stats (game_id, player_id, team_id, minutes, pts, reb, ast, stl, blk, tov)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = ("0022500001", 10, 1, 32.5, 20, 8, 5, 1, 2, 3)
    with seeded_postgres.transaction():
        seeded_postgres.execute(statement, values)
        seeded_postgres.execute("INSERT INTO players (player_id, name) VALUES (%s, %s)", (11, "Second Player"))
        seeded_postgres.execute(statement, (values[0], 11, *values[2:]))
    assert seeded_postgres.execute("SELECT COUNT(*) FROM player_game_stats").fetchone() == (2,)
    with pytest.raises(psycopg.errors.UniqueViolation):
        with seeded_postgres.transaction():
            seeded_postgres.execute(statement, values)

def test_postgres_schema_rejects_missing_minutes(seeded_postgres):
    with pytest.raises(psycopg.errors.NotNullViolation):
        with seeded_postgres.transaction():
            seeded_postgres.execute(
                "INSERT INTO player_game_stats (game_id, player_id, team_id, minutes, pts, reb, ast, stl, blk, tov) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                ("0022500001", 10, 1, None, 20, 8, 5, 1, 2, 3))