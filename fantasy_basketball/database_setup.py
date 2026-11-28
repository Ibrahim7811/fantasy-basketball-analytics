import sqlite3
from pathlib import Path
from fantasy_basketball.migration import migrate_sqlite_to_postgres
from fantasy_basketball.postgres_database import get_postgres_connection
import argparse
import json
import os
import psycopg

def seed_postgres_from_sqlite(sqlite_path: str, database_url: str) -> dict[str, int]:
    source_path = Path(sqlite_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError("Source SQLite database does not exist")
    if not database_url.strip():
        raise ValueError("PostgreSQL connection settings are required")
    source = sqlite3.connect(source_path.as_uri() + "?mode=ro", uri = True)
    try:
        source.execute("BEGIN")
        with get_postgres_connection(database_url) as destination:
            counts = migrate_sqlite_to_postgres(source, destination)
        return counts
    finally:
        source.close()

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description = "Seed an empty PostgreSQL database from SQLite")
    parser.add_argument("--sqlite-path", required = True, help = "Path to the source SQLite database")
    args = parser.parse_args(argv)
    database_url = os.environ.get("FANTASY_DATABASE_URL", "").strip()
    if not database_url:
        parser.error("Set FANTASY_DATABASE_URL before running database setup")
    try:
        counts = seed_postgres_from_sqlite(args.sqlite_path, database_url)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(status = 2, message = f"Database setup rejected: {error}\n")
    except (sqlite3.Error, psycopg.Error, OSError):
        parser.exit(status = 1, message = "Database setup failed. Check database connectivity, permissions, and source data.\n")
    print("Database setup completed successfully.")
    print(json.dumps(counts, indent = 2, sort_keys = True))

if __name__ == "__main__":
    main()