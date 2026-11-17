import sqlite3

def get_connection(db_path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection

def initialize_database(connection: sqlite3.Connection) -> None:
    connection.execute("""
    CREATE TABLE IF NOT EXISTS teams (
        team_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        abbreviation TEXT NOT NULL UNIQUE
    )""")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        )""")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            season TEXT NOT NULL,
            home_team INTEGER NOT NULL REFERENCES teams(team_id),
            away_team INTEGER NOT NULL REFERENCES teams(team_id),
            CHECK (home_team <> away_team)
        )""")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS player_game_stats (
            game_id TEXT NOT NULL REFERENCES games(game_id),
            player_id INTEGER NOT NULL REFERENCES players(player_id),
            team_id INTEGER NOT NULL REFERENCES teams(team_id),
            minutes REAL NOT NULL,
            pts INTEGER NOT NULL,
            reb INTEGER NOT NULL,
            ast INTEGER NOT NULL,
            stl INTEGER NOT NULL,
            blk INTEGER NOT NULL,
            tov INTEGER NOT NULL,
            PRIMARY KEY (game_id, player_id)
        )""")

def upsert_team(connection: sqlite3.Connection, team_id: int, name: str, abbreviation: str,) -> None:
    connection.execute("""
    INSERT INTO teams (team_id, name, abbreviation)
    VALUES (?, ?, ?)
    ON CONFLICT(team_id) DO UPDATE SET
        name = excluded.name,
        abbreviation = excluded.abbreviation
    """, (team_id, name, abbreviation))