import pytest
from data_loader import load_players
from lineup import build_optimal_lineup, lineup_total


def test_csv_to_optimal_lineup(tmp_path):
    # Create a temporary CSV file
    csv_file = tmp_path / "players.csv"

    # Write sample player information
    csv_file.write_text(
        "name,positions,projected_points,schedule,team\n"
        "Player A,PG;SG,60,2026-11-09,DEN\n"
        "Player B,PG,59,2026-11-09,BOS\n"
        "Player C,SG,55,2026-11-09,DET\n",
        encoding="utf-8"
    )

    # Load Player objects from the CSV
    players = load_players(csv_file)

    # Define lineup settings
    date = "2026-11-09"
    lineup_slots = ["PG", "SG"]

    # Generate the optimal lineup
    optimal_lineup = build_optimal_lineup(
        players, date, lineup_slots
    )

    # Verify CSV data was loaded
    assert len(players) == 3

    # Verify both lineup slots are filled
    assert len(optimal_lineup) == 2

    # Verify the optimal player assignments
    assert optimal_lineup["PG"]._name == "Player B"
    assert optimal_lineup["SG"]._name == "Player A"

    # Verify no player was assigned twice
    assert len(set(optimal_lineup.values())) == 2

    # Verify positional eligibility
    for slot, player in optimal_lineup.items():
        assert player.eligible_for(slot)
        assert player.plays_on(date)

    # Verify the optimal projected total
    assert lineup_total(optimal_lineup) == pytest.approx(119.0)
