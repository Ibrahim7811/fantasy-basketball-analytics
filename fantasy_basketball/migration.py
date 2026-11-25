import sqlite3
from datetime import date
import psycopg
from psycopg import sql
from fantasy_basketball.postgres_database import initialize_postgres_database

def migrate_sqlite_to_postgres(sqlite_connection: sqlite3.Connection, postgres_connection: psycopg.Connection,) -> dict[str, int]:
    tables = {
        "teams": ("team_id", "name", "abbreviation"),
        "players": ("player_id", "name"),
        "games": ("game_id", "date", "season", "home_team", "away_team"),
        "player_game_stats": (
            "game_id", "player_id", "team_id", "minutes",
            "pts", "reb", "ast", "stl", "blk", "tov")}
    counts = {}
    with postgres_connection.transaction():
        initialize_postgres_database(postgres_connection)
        for table in tables:
            query = sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table))
            existing_count = postgres_connection.execute(query).fetchone()[0]
            if existing_count:
                raise ValueError(f"Destination table {table} must be empty")
        with postgres_connection.cursor() as cursor:
            for table, columns in tables.items():
                # Table and column names come only from the fixed dictionary above.
                source_query = f"SELECT {', '.join(columns)} FROM {table}"
                rows = sqlite_connection.execute(source_query).fetchall()
                if table == "games":
                    rows = [(row[0], date.fromisoformat(row[1]), *row[2:]) for row in rows]
                insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                    sql.Identifier(table),
                    sql.SQL(", ").join(sql.Identifier(column) for column in columns),
                    sql.SQL(", ").join(sql.Placeholder() for _ in columns))
                if rows:
                    cursor.executemany(insert_query, rows)
                counts[table] = len(rows)
        for table, expected_count in counts.items():
            query = sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table))
            actual_count = postgres_connection.execute(query).fetchone()[0]
            if actual_count != expected_count:
                raise ValueError(f"Row count mismatch for {table}")
    return counts