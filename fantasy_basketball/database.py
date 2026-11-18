import sqlite3
import pandas as pd

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

def upsert_player(connection: sqlite3.Connection, player_id: int, name: str,) -> None:
    connection.execute("""
    INSERT INTO players (player_id, name)
    VALUES (?, ?)
    ON CONFLICT(player_id) DO UPDATE SET
        name = excluded.name
    """, (player_id, name))

def upsert_game(connection: sqlite3.Connection, game_id: str, game_date: str, season: str, home_team: int, away_team: int,) -> None:
    connection.execute("""
        INSERT INTO games (game_id, date, season, home_team, away_team)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(game_id) DO UPDATE SET
            date = excluded.date,
            season = excluded.season,
            home_team = excluded.home_team,
            away_team = excluded.away_team
        """, (game_id, game_date, season, home_team, away_team))

def upsert_player_game_stats(
    connection: sqlite3.Connection,
    game_id: str,
    player_id: int,
    team_id: int,
    minutes: float,
    pts: int,
    reb: int,
    ast: int,
    stl: int,
    blk: int,
    tov: int,) -> None:
    connection.execute("""
        INSERT INTO player_game_stats (game_id, player_id, team_id, minutes, pts, reb, ast, stl, blk, tov)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(game_id, player_id) DO UPDATE SET
            team_id = excluded.team_id,
            minutes = excluded.minutes,
            pts = excluded.pts,
            reb = excluded.reb,
            ast = excluded.ast,
            stl = excluded.stl,
            blk = excluded.blk,
            tov = excluded.tov
        """, (game_id, player_id, team_id, minutes, pts, reb, ast, stl, blk, tov))

def load_player_game_logs(connection: sqlite3.Connection, player_id: int, season: str,) -> pd.DataFrame:
    query = """
        SELECT
            s.game_id AS GAME_ID,
            s.player_id AS PLAYER_ID,
            s.team_id AS TEAM_ID,
            g.date AS GAME_DATE,
            s.minutes AS MIN,
            s.pts AS PTS,
            s.reb AS REB,
            s.ast AS AST,
            s.stl AS STL,
            s.blk AS BLK,
            s.tov AS TOV
        FROM player_game_stats AS s
        JOIN games AS g ON s.game_id = g.game_id
        WHERE s.player_id = ?
          AND g.season = ?
        ORDER BY g.date ASC, g.game_id ASC
    """
    return pd.read_sql_query(query,connection,params=(player_id, season),parse_dates=["GAME_DATE"],)