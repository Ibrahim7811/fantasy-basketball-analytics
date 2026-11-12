
import pytest

from player_class import Player
from data_loader import load_players


def test_load_players(tmp_path):
    csv_file = tmp_path / "players.csv"

    csv_file.write_text(
        "name,positions,projected_points,schedule,team\n"
        "Nikola Jokic,C,58.5,2026-11-05,DEN\n"
        "Cade Cunningham,PG; SG,48.25,2026-11-05; 2026-11-07,DET\n",
        encoding="utf-8"
    )

    players = load_players(csv_file)

    # Verify two Player objects were loaded
    assert len(players) == 2
    assert all(isinstance(player, Player) for player in players)

    # Verify Nikola Jokic's information
    assert players[0]._name == "Nikola Jokic"
    assert players[0]._positions == ["C"]
    assert players[0]._projected_points == pytest.approx(58.5)
    assert isinstance(players[0]._projected_points, float)
    assert players[0]._team == "DEN"

    # Verify Cade Cunningham's information
    assert players[1]._name == "Cade Cunningham"
    assert players[1]._positions == ["PG", "SG"]
    assert players[1]._projected_points == pytest.approx(48.25)
    assert players[1]._schedule == ["2026-11-05", "2026-11-07"]

    # Verify schedule functionality
    assert players[1].plays_on("2026-11-07")
    assert not players[1].plays_on("2026-11-06")

def test_load_empty_csv(tmp_path):
    # Create a temporary CSV file
    csv_file = tmp_path / "players.csv"

    # Write only the column headers (no player records)
    csv_file.write_text(
        "name,positions,projected_points,schedule,team\n",
        encoding="utf-8"
    )

    # Load players from the empty CSV
    players = load_players(csv_file)

    # Verify that an empty list is returned
    assert players == []
    assert len(players) == 0

def test_invalid_projection(tmp_path):
    # Create a temporary CSV file
    csv_file = tmp_path / "players.csv"

    # Write a player with an invalid projection
    csv_file.write_text(
        "name,positions,projected_points,schedule,team\n"
        "Nikola Jokic,C,not_a_number,2026-11-05,DEN\n",
        encoding="utf-8"
    )

    # Verify that loading the CSV raises ValueError
    with pytest.raises(ValueError):
        load_players(csv_file)

def test_missing_required_column(tmp_path):
    # Create a temporary CSV file
    csv_file = tmp_path / "players.csv"

    # Create CSV data without the projected_points column
    csv_file.write_text(
        "name,positions,schedule,team\n"
        "Nikola Jokic,C,2026-11-05,DEN\n",
        encoding="utf-8"
    )

    # Verify that loading the CSV raises KeyError
    with pytest.raises(ValueError, match="projected_points"):
        load_players(csv_file)

def test_empty_schedule(tmp_path):
    # Create a temporary CSV file
    csv_file = tmp_path / "players.csv"

    # Write a player with an empty schedule field
    csv_file.write_text(
        "name,positions,projected_points,schedule,team\n"
        "Nikola Jokic,C,58.5,,DEN\n",
        encoding="utf-8"
    )

    # Load players from the CSV
    players = load_players(csv_file)

    # Verify one player was loaded
    assert len(players) == 1

    # Verify the schedule is an empty list
    assert players[0]._schedule == []

    # Verify the player has no scheduled games
    assert not players[0].plays_on("2026-11-05")
    assert not players[0].plays_on("2026-11-07")

def test_missing_headers_empty_file(tmp_path):
    # Create a temporary CSV file
    csv_file = tmp_path / "players.csv"

    # Create a completely empty file
    csv_file.write_text("", encoding="utf-8")

    # Verify that loading the file raises ValueError
    with pytest.raises(ValueError, match="Missing required columns"):
        load_players(csv_file)

def test_empty_player_name(tmp_path):
    # Create a temporary CSV file
    csv_file = tmp_path / "players.csv"

    # Create a player record with an empty name
    csv_file.write_text(
        "name,positions,projected_points,schedule,team\n"
        ",PG;SG,48.25,2026-11-09,DET\n",
        encoding="utf-8"
    )

    # Verify that an empty name raises ValueError
    with pytest.raises(ValueError, match="Missing player name"):
        load_players(csv_file)

def test_empty_player_positions(tmp_path):
    csv_file = tmp_path / "players.csv"

    # Create a player with no eligible positions
    csv_file.write_text(
        "name,positions,projected_points,schedule,team\n"
        "Nikola Jokic,,58.5,2026-11-09,DEN\n",
        encoding="utf-8"
    )

    # Verify that empty positions raise ValueError
    with pytest.raises(ValueError, match="Invalid player positions"):
        load_players(csv_file)

def test_invalid_player_positions(tmp_path):
    csv_file = tmp_path / "players.csv"

    # Create a player with one valid and one invalid position
    csv_file.write_text(
        "name,positions,projected_points,schedule,team\n"
        "Cade Cunningham,PG;XYZ,48.25,2026-11-09,DET\n",
        encoding="utf-8"
    )

    # Verify that unsupported positions raise ValueError
    with pytest.raises(ValueError, match="Invalid player positions"):
        load_players(csv_file)

@pytest.mark.parametrize(
    "invalid_projection",
    [
        "",
        "abc",
        "NaN",
        "inf",
        "-inf",
        "   "
    ]
)
def test_invalid_projected_points(tmp_path, invalid_projection):
    csv_file = tmp_path / "players.csv"

    csv_file.write_text(
        "name,positions,projected_points,schedule,team\n"
        f"Nikola Jokic,C,{invalid_projection},2026-11-09,DEN\n",
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Invalid projected points"):
        load_players(csv_file)

@pytest.mark.parametrize(
    "projection, expected",
    [
        ("58.5", 58.5),
        ("0", 0.0),
        ("-10.5", -10.5),
        ("125.75", 125.75),
        (" 42.5 ", 42.5)
    ]
)
def test_valid_projected_points(tmp_path, projection, expected):
    # Create a temporary CSV file
    csv_file = tmp_path / "players.csv"

    # Write a player with the specified projection
    csv_file.write_text(
        "name,positions,projected_points,schedule,team\n"
        f"Nikola Jokic,C,{projection},2026-11-09,DEN\n",
        encoding="utf-8"
    )

    # Load players
    players = load_players(csv_file)

    # Verify one player was loaded
    assert len(players) == 1

    # Verify projected points are stored as a float
    assert isinstance(players[0]._projected_points, float)

    # Verify the projection matches the expected value
    assert players[0]._projected_points == pytest.approx(expected)

def test_invalid_nba_team(tmp_path):
    csv_file = tmp_path / "players.csv"

    # Create a player with a nonexistent NBA team
    csv_file.write_text(
        "name,positions,projected_points,schedule,team\n"
        "Nikola Jokic,C,58.5,2026-11-09,XYZ\n",
        encoding="utf-8"
    )

    # Verify that an invalid team raises ValueError
    with pytest.raises(ValueError, match="Invalid NBA team"):
        load_players(csv_file)

def test_valid_nba_team_lowercase(tmp_path):
    csv_file = tmp_path / "players.csv"

    # Create a player with a lowercase team abbreviation
    csv_file.write_text(
        "name,positions,projected_points,schedule,team\n"
        "Nikola Jokic,C,58.5,2026-11-09,den\n",
        encoding="utf-8"
    )

    # Load the player
    players = load_players(csv_file)

    # Verify the player was loaded
    assert len(players) == 1

    # Verify the team abbreviation was normalized
    assert players[0]._team == "DEN"

@pytest.mark.parametrize(
    "data_row",
    [
        # Missing team field (4 fields instead of 5)
        "Nikola Jokic,C,58.5,2026-11-09\n",

        # Extra field (6 fields instead of 5)
        "Nikola Jokic,C,58.5,2026-11-09,DEN,EXTRA\n"
    ]
)
def test_malformed_csv_row(tmp_path, data_row):
    csv_file = tmp_path / "players.csv"

    csv_file.write_text(
        "name,positions,projected_points,schedule,team\n"
        + data_row,
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Malformed CSV row"):
        load_players(csv_file)
