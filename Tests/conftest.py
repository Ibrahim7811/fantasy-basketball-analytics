import os
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4
import pytest
from dotenv import dotenv_values
from psycopg import sql
from fantasy_basketball.postgres_database import get_postgres_connection, initialize_postgres_database

@pytest.fixture
def postgres_connection():
    database_url = os.environ.get("TEST_POSTGRES_URL")
    if not database_url:
        if os.environ.get("RUN_POSTGRES_TESTS") != "1":
            pytest.skip("Set RUN_POSTGRES_TESTS=1 to enable PostgreSQL integration tests")
        project_root = Path(__file__).resolve().parent.parent
        password = dotenv_values(project_root / ".env").get("POSTGRES_PASSWORD")
        if not password:
            pytest.fail("POSTGRES_PASSWORD is missing from .env")
        database_url = (f"postgresql://fantasy_app:{quote(password, safe='')}""@127.0.0.1:5433/fantasy_basketball")
    connection = get_postgres_connection(database_url)
    schema_name = f"test_{uuid4().hex}"
    try:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name)))
        connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema_name)))
        connection.commit()
        with connection.transaction():
            initialize_postgres_database(connection)
        yield connection
    finally:
        try:
            connection.rollback()
            connection.execute("SET search_path TO public")
            connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema_name)))
            connection.commit()
        finally:
            connection.close()