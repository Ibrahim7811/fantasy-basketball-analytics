import psycopg
import pandas as pd

def get_postgres_connection(database_url: str) -> psycopg.Connection:
    return psycopg.connect(database_url, connect_timeout=10)

def initialize_postgres_database(connection: psycopg.Connection) -> None:
    connection.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            abbreviation TEXT NOT NULL UNIQUE
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            date DATE NOT NULL,
            season TEXT NOT NULL,
            home_team INTEGER NOT NULL REFERENCES teams(team_id),
            away_team INTEGER NOT NULL REFERENCES teams(team_id),
            CHECK (home_team <> away_team)
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS player_game_stats (
            game_id TEXT NOT NULL REFERENCES games(game_id),
            player_id INTEGER NOT NULL REFERENCES players(player_id),
            team_id INTEGER NOT NULL REFERENCES teams(team_id),
            minutes DOUBLE PRECISION NOT NULL,
            pts INTEGER NOT NULL,
            reb INTEGER NOT NULL,
            ast INTEGER NOT NULL,
            stl INTEGER NOT NULL,
            blk INTEGER NOT NULL,
            tov INTEGER NOT NULL,
            PRIMARY KEY (game_id, player_id)
        )
    """)

def load_postgres_player_game_logs(connection: psycopg.Connection, player_id: int, season: str) -> pd.DataFrame:
    query = """
        SELECT
            s.game_id AS "GAME_ID",
            s.player_id AS "PLAYER_ID",
            s.team_id AS "TEAM_ID",
            g.date AS "GAME_DATE",
            s.minutes AS "MIN",
            s.pts AS "PTS",
            s.reb AS "REB",
            s.ast AS "AST",
            s.stl AS "STL",
            s.blk AS "BLK",
            s.tov AS "TOV"
        FROM player_game_stats AS s
        JOIN games AS g ON g.game_id = s.game_id
        WHERE s.player_id = %s AND g.season = %s
        ORDER BY g.date, g.game_id
    """
    with connection.cursor() as cursor:
        cursor.execute(query, (player_id, season))
        rows = cursor.fetchall()
        columns = [column.name for column in cursor.description]
    logs = pd.DataFrame(rows, columns=columns)
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    return logs