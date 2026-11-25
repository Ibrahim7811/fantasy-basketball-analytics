from unittest.mock import Mock
import pytest
import psycopg
from fantasy_basketball import postgres_database

def test_get_postgres_connection_returns_connection(monkeypatch):
    connection = Mock()
    connector = Mock(return_value=connection)
    monkeypatch.setattr(postgres_database.psycopg, "connect", connector)
    database_url = "postgresql://test_user:test_password@localhost:5433/test_db"
    result = postgres_database.get_postgres_connection(database_url)
    assert result is connection
    connector.assert_called_once_with(database_url, connect_timeout=10)

def test_get_postgres_connection_propagates_connection_error(monkeypatch):
    connector = Mock(side_effect=psycopg.OperationalError("Connection failed"))
    monkeypatch.setattr(postgres_database.psycopg, "connect", connector)
    with pytest.raises(psycopg.OperationalError, match="Connection failed"):
        postgres_database.get_postgres_connection("postgresql://localhost:5433/test_db")