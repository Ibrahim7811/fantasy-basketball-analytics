import pandas as pd
import pytest
from fantasy_basketball.postgres_database import load_postgres_player_game_logs

@pytest.fixture
def player_logs_database(postgres_connection):
    with postgres_connection.transaction():
        postgres_connection.execute("INSERT INTO teams (team_id, name, abbreviation) VALUES (1, 'Home', 'HOM'), (2, 'Away', 'AWY')")
        postgres_connection.execute("INSERT INTO players (player_id, name) VALUES (10, 'First Player'), (20, 'Second Player')")
        with postgres_connection.cursor() as cursor:
            cursor.executemany("INSERT INTO games (game_id, date, season, home_team, away_team) VALUES (%s, %s, %s, %s, %s)", [
                ("0022500002", "2026-02-05", "2025-26", 2, 1),
                ("0022500001", "2026-02-03", "2025-26", 1, 2),
                ("0022400001", "2025-02-03", "2024-25", 1, 2)])
            cursor.executemany("INSERT INTO player_game_stats (game_id, player_id, team_id, minutes, pts, reb, ast, stl, blk, tov) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", [
                ("0022500002", 10, 1, 35.25, 30, 9, 6, 2, 3, 4),
                ("0022500001", 10, 1, 32.5, 20, 8, 5, 1, 2, 3),
                ("0022500001", 20, 2, 30.5, 18, 6, 4, 2, 1, 2),
                ("0022400001", 10, 1, 28.0, 15, 7, 3, 0, 1, 1)])
    return postgres_connection

def test_load_postgres_logs_returns_records_in_date_order(player_logs_database):
    logs = load_postgres_player_game_logs(player_logs_database, 10, "2025-26")
    assert list(logs.columns) == ["GAME_ID", "PLAYER_ID", "TEAM_ID", "GAME_DATE", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV"]
    assert list(logs.itertuples(index=False, name=None)) == [
        ("0022500001", 10, 1, pd.Timestamp("2026-02-03"), 32.5, 20, 8, 5, 1, 2, 3),
        ("0022500002", 10, 1, pd.Timestamp("2026-02-05"), 35.25, 30, 9, 6, 2, 3, 4)]
    assert pd.api.types.is_datetime64_any_dtype(logs["GAME_DATE"])

def test_load_postgres_logs_filters_by_player(player_logs_database):
    logs = load_postgres_player_game_logs(player_logs_database, 20, "2025-26")
    assert len(logs) == 1
    assert logs["PLAYER_ID"].tolist() == [20]
    assert logs["PTS"].tolist() == [18]

def test_load_postgres_logs_filters_by_season(player_logs_database):
    logs = load_postgres_player_game_logs(player_logs_database, 10, "2024-25")
    assert logs["GAME_ID"].tolist() == ["0022400001"]
    assert logs["GAME_DATE"].tolist() == [pd.Timestamp("2025-02-03")]
    assert logs["PTS"].tolist() == [15]

@pytest.mark.parametrize("player_id, season", [(999, "2025-26"), (10, "1999-00")])
def test_load_postgres_logs_preserves_columns_when_empty(player_logs_database, player_id, season):
    logs = load_postgres_player_game_logs(player_logs_database, player_id, season)
    assert logs.empty
    assert list(logs.columns) == ["GAME_ID", "PLAYER_ID", "TEAM_ID", "GAME_DATE", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV"]
    assert pd.api.types.is_datetime64_any_dtype(logs["GAME_DATE"])