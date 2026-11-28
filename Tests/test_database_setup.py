import sqlite3
from unittest.mock import MagicMock, Mock
import pytest
from fantasy_basketball import database_setup
import json
import psycopg

@pytest.fixture
def seed_source(tmp_path):
    path = tmp_path / "source.db"
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE marker (value TEXT)")
        connection.execute("INSERT INTO marker VALUES (?)", ("preserve",))
        connection.commit()
    finally:
        connection.close()
    return path

@pytest.fixture
def seed_dependencies(seed_source, monkeypatch):
    source = sqlite3.connect(seed_source.as_uri() + "?mode=ro", uri = True)
    sqlite_connector = Mock(return_value = source)
    destination = MagicMock()
    destination.__enter__.return_value = destination
    postgres_connector = Mock(return_value = destination)
    counts = {"teams": 30, "players": 32, "games": 2460, "player_game_stats": 4118}
    migrator = Mock(return_value = counts)
    monkeypatch.setattr(database_setup.sqlite3, "connect", sqlite_connector)
    monkeypatch.setattr(database_setup, "get_postgres_connection", postgres_connector)
    monkeypatch.setattr(database_setup, "migrate_sqlite_to_postgres", migrator)
    try:
        yield {
            "source": source,
            "sqlite_connector": sqlite_connector,
            "destination": destination,
            "postgres_connector": postgres_connector,
            "migrator": migrator,
            "counts": counts}
    finally:
        source.close()

def test_seed_postgres_returns_counts_and_closes_connections(seed_source, seed_dependencies):
    dependencies = seed_dependencies
    database_url = "postgresql://localhost/fantasy_test"
    result = database_setup.seed_postgres_from_sqlite(str(seed_source), database_url)
    assert result == dependencies["counts"]
    dependencies["sqlite_connector"].assert_called_once_with(seed_source.as_uri() + "?mode=ro", uri = True)
    dependencies["postgres_connector"].assert_called_once_with(database_url)
    dependencies["migrator"].assert_called_once_with(dependencies["source"], dependencies["destination"])
    dependencies["destination"].__exit__.assert_called_once_with(None, None, None)
    with pytest.raises(sqlite3.ProgrammingError, match = "closed"):
        dependencies["source"].execute("SELECT 1")

def test_seed_postgres_uses_readonly_snapshot(seed_source, seed_dependencies):
    dependencies = seed_dependencies
    def inspect_source(source, destination):
        assert source.in_transaction
        assert source.execute("SELECT value FROM marker").fetchall() == [("preserve",)]
        with pytest.raises(sqlite3.OperationalError, match = "readonly"):
            source.execute("INSERT INTO marker VALUES ('changed')")
        return dependencies["counts"]
    dependencies["migrator"].side_effect = inspect_source
    result = database_setup.seed_postgres_from_sqlite(str(seed_source), "postgresql://localhost/fantasy_test")
    assert result == dependencies["counts"]

def test_seed_postgres_closes_source_when_connection_fails(seed_source, seed_dependencies):
    dependencies = seed_dependencies
    dependencies["postgres_connector"].side_effect = RuntimeError("Connection failed")
    with pytest.raises(RuntimeError, match = "Connection failed"):
        database_setup.seed_postgres_from_sqlite(str(seed_source), "postgresql://localhost/fantasy_test")
    dependencies["migrator"].assert_not_called()
    with pytest.raises(sqlite3.ProgrammingError, match = "closed"):
        dependencies["source"].execute("SELECT 1")

def test_seed_postgres_propagates_migration_failure_and_closes_connections(seed_source, seed_dependencies):
    dependencies = seed_dependencies
    error = ValueError("Destination table teams must be empty")
    dependencies["migrator"].side_effect = error
    with pytest.raises(ValueError, match = "must be empty"):
        database_setup.seed_postgres_from_sqlite(str(seed_source), "postgresql://localhost/fantasy_test")
    exit_args = dependencies["destination"].__exit__.call_args.args
    assert exit_args[0] is ValueError
    assert exit_args[1] is error
    with pytest.raises(sqlite3.ProgrammingError, match = "closed"):
        dependencies["source"].execute("SELECT 1")

def test_seed_postgres_rejects_missing_source_without_creating_it(tmp_path, monkeypatch):
    source_path = tmp_path / "missing.db"
    connector = Mock()
    monkeypatch.setattr(database_setup, "get_postgres_connection", connector)
    with pytest.raises(FileNotFoundError, match = "Source SQLite database does not exist"):
        database_setup.seed_postgres_from_sqlite(str(source_path), "postgresql://localhost/fantasy_test")
    assert not source_path.exists()
    connector.assert_not_called()

@pytest.mark.parametrize("database_url", ["", " ", "\t\n"])
def test_seed_postgres_rejects_blank_connection_settings(seed_source, monkeypatch, database_url):
    connector = Mock()
    monkeypatch.setattr(database_setup, "get_postgres_connection", connector)
    with pytest.raises(ValueError, match = "connection settings are required"):
        database_setup.seed_postgres_from_sqlite(str(seed_source), database_url)
    connector.assert_not_called()

def test_database_setup_main_runs_seed_and_prints_counts(monkeypatch, capsys):
    database_url = "postgresql://localhost/fantasy_test"
    counts = {"teams": 30, "players": 32, "games": 2460, "player_game_stats": 4118}
    seeder = Mock(return_value=counts)
    monkeypatch.setenv("FANTASY_DATABASE_URL", database_url)
    monkeypatch.setattr(database_setup, "seed_postgres_from_sqlite", seeder)
    database_setup.main(["--sqlite-path", "source.db"])
    seeder.assert_called_once_with("source.db", database_url)
    output = capsys.readouterr()
    first_line, counts_json = output.out.split("\n", 1)
    assert first_line == "Database setup completed successfully."
    assert json.loads(counts_json) == counts
    assert output.err == ""

@pytest.mark.parametrize("value", [None, "", " \t"])
def test_database_setup_main_requires_connection_settings(monkeypatch, capsys, value):
    if value is None:
        monkeypatch.delenv("FANTASY_DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("FANTASY_DATABASE_URL", value)
    seeder = Mock()
    monkeypatch.setattr(database_setup, "seed_postgres_from_sqlite", seeder)
    with pytest.raises(SystemExit) as caught:
        database_setup.main(["--sqlite-path", "source.db"])
    assert caught.value.code == 2
    assert "Set FANTASY_DATABASE_URL" in capsys.readouterr().err
    seeder.assert_not_called()

def test_database_setup_main_requires_source_argument(monkeypatch, capsys):
    monkeypatch.setenv("FANTASY_DATABASE_URL", "postgresql://localhost/fantasy_test")
    seeder = Mock()
    monkeypatch.setattr(database_setup, "seed_postgres_from_sqlite", seeder)
    with pytest.raises(SystemExit) as caught:
        database_setup.main([])
    assert caught.value.code == 2
    assert "--sqlite-path" in capsys.readouterr().err
    seeder.assert_not_called()

def test_database_setup_main_help_does_not_connect(monkeypatch, capsys):
    monkeypatch.delenv("FANTASY_DATABASE_URL", raising=False)
    seeder = Mock()
    monkeypatch.setattr(database_setup, "seed_postgres_from_sqlite", seeder)
    with pytest.raises(SystemExit) as caught:
        database_setup.main(["--help"])
    assert caught.value.code == 0
    assert "Seed an empty PostgreSQL database" in capsys.readouterr().out
    seeder.assert_not_called()

@pytest.mark.parametrize("error", [
    ValueError("Destination table teams must be empty"),
    FileNotFoundError("Source SQLite database does not exist")
])
def test_database_setup_main_reports_rejected_setup(monkeypatch, capsys, error):
    monkeypatch.setenv("FANTASY_DATABASE_URL", "postgresql://localhost/fantasy_test")
    monkeypatch.setattr(database_setup, "seed_postgres_from_sqlite", Mock(side_effect=error))
    with pytest.raises(SystemExit) as caught:
        database_setup.main(["--sqlite-path", "source.db"])
    assert caught.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert f"Database setup rejected: {error}" in output.err

@pytest.mark.parametrize("error", [
    sqlite3.OperationalError("Private source details"),
    psycopg.OperationalError("Private connection details"),
    OSError("Private filesystem details")
])
def test_database_setup_main_handles_operational_failure(monkeypatch, capsys, error):
    monkeypatch.setenv("FANTASY_DATABASE_URL", "postgresql://localhost/fantasy_test")
    monkeypatch.setattr(database_setup, "seed_postgres_from_sqlite", Mock(side_effect=error))
    with pytest.raises(SystemExit) as caught:
        database_setup.main(["--sqlite-path", "source.db"])
    assert caught.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Database setup failed." in output.err
    assert "Private" not in output.err