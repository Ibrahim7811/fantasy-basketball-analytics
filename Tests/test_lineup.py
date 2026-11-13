import pytest
from fantasy_basketball.player_class import Player
from fantasy_basketball.lineup import build_greedy_lineup, build_optimal_lineup, lineup_total

@pytest.fixture
def sample_roster():
    """Create a fresh roster and lineup configuration for each test."""
    players = [
        Player("Nikola Jokic", ["C"], 58.5, ["2026-11-05"], "DEN"),
        Player("Victor Wembanyama", ["C", "PF"], 55.2, ["2026-11-05"], "SAS"),
        Player("Cade Cunningham", ["PG", "SG"], 48.25, ["2026-11-05"], "DET"),
        Player("Jalen Brunson", ["PG"], 46.8, ["2026-11-05"], "NYK"),
        Player("Anthony Edwards", ["SG", "SF"], 47.5, ["2026-11-05"], "MIN"),
        Player("Paolo Banchero", ["PF", "SF"], 42.5, ["2026-11-05"], "ORL"),
        Player("Inactive Guard", ["PG"], 100.0, ["2026-11-06"], "TEST"),
    ]
    slots = ["PG", "SG", "C1", "C2", "UTIL1"]
    return players, slots

def test_fills_available_slots(sample_roster):
    players, slots = sample_roster
    lineup = build_greedy_lineup(players, "2026-11-05", slots)

    assert isinstance(lineup, dict)
    assert set(lineup) == set(slots)

def test_no_duplicate_players(sample_roster):
    players, slots = sample_roster
    lineup = build_greedy_lineup(players, "2026-11-05", slots)

    assigned_players = list(lineup.values())
    assert len(assigned_players) == len(set(assigned_players))

def test_only_players_with_a_game_are_selected(sample_roster):
    players, slots = sample_roster
    date = "2026-11-05"
    lineup = build_greedy_lineup(players, date, slots)

    assert all(player.plays_on(date) for player in lineup.values())
    assert all(player._name != "Inactive Guard" for player in lineup.values())

def test_every_player_is_eligible_for_assigned_slot(sample_roster):
    players, slots = sample_roster
    lineup = build_greedy_lineup(players, "2026-11-05", slots)

    assert all(player.eligible_for(slot) for slot, player in lineup.items())

def test_picks_highest_projected_player_for_one_slot(sample_roster):
    players, _ = sample_roster
    lineup = build_greedy_lineup(players, "2026-11-05", ["PG"])

    assert lineup["PG"]._name == "Cade Cunningham"

def test_returns_empty_lineup_when_nobody_plays(sample_roster):
    players, slots = sample_roster
    lineup = build_greedy_lineup(players, "2026-11-07", slots)

    assert lineup == {}

def test_calls_do_not_share_selection_state(sample_roster):
    players, slots = sample_roster
    first = build_greedy_lineup(players, "2026-11-05", slots)
    second = build_greedy_lineup(players, "2026-11-05", slots)

    assert first == second
    assert first is not second

def test_lineup_projected_total(sample_roster):
    players, slots = sample_roster
    lineup = build_greedy_lineup(players, "2026-11-05", slots)

    total = sum(player._projected_points for player in lineup.values())
    assert total == pytest.approx(256.25)

def test_lineup_total():
    jokic = Player("Nikola Jokic", ["C"], 58.5, ["2026-11-05"], "DEN")
    cade = Player("Cade Cunningham", ["PG", "SG"], 48.25, ["2026-11-05"], "DET")

    # Test 1: Empty lineup
    assert lineup_total({}) == pytest.approx(0.0)

    # Test 2: One player
    lineup = {"C1": jokic}
    assert lineup_total(lineup) == pytest.approx(58.5)

    # Test 3: Two players
    lineup = {"C1": jokic, "PG": cade}
    assert lineup_total(lineup) == pytest.approx(106.75)

def test_optimal_beats_greedy():
    players = [
        Player("Player A", ["PG", "SG"], 60.0, ["2026-11-05"], "AAA"),
        Player("Player B", ["PG"], 59.0, ["2026-11-05"], "BBB"),
        Player("Player C", ["SG"], 55.0, ["2026-11-05"], "CCC")
    ]

    lineup_slots = ["PG", "SG"]
    date = "2026-11-05"

    greedy_lineup = build_greedy_lineup(players, date, lineup_slots)
    optimal_lineup = build_optimal_lineup(players, date, lineup_slots)

    greedy_total = lineup_total(greedy_lineup)
    optimal_total = lineup_total(optimal_lineup)

    assert greedy_total == pytest.approx(115.0)
    assert optimal_total == pytest.approx(119.0)
    assert optimal_total > greedy_total

    # Verify the optimal player assignments
    assert optimal_lineup["PG"]._name == "Player B"
    assert optimal_lineup["SG"]._name == "Player A"

def test_optimal_with_unfillable_slot():
    players = [
        Player("Player A", ["PG"], 50.0, ["2026-11-05"], "AAA"),
        Player("Player B", ["C"], 40.0, ["2026-11-05"], "BBB")
    ]

    lineup_slots = ["PG", "SF", "C1"]
    date = "2026-11-05"

    optimal_lineup = build_optimal_lineup(players, date, lineup_slots)

    # Verify the correct players are assigned
    assert optimal_lineup["PG"]._name == "Player A"
    assert optimal_lineup["C1"]._name == "Player B"

    # Verify the SF slot remains empty
    assert "SF" not in optimal_lineup

    # Verify exactly two players are selected
    assert len(optimal_lineup) == 2

    # Verify the total projected points
    assert lineup_total(optimal_lineup) == pytest.approx(90.0)

def test_optimal_negative_projections():
    players = [
        Player("Player A", ["PG"], 60.0, ["2026-11-05"], "AAA"),
        Player("Player B", ["SG"], 45.0, ["2026-11-05"], "BBB"),
        Player("Player C", ["C"], -10.0, ["2026-11-05"], "CCC")
    ]

    lineup_slots = ["PG", "SG", "C1"]
    date = "2026-11-05"

    optimal_lineup = build_optimal_lineup(players, date, lineup_slots)

    # Verify positive-projection players are selected
    assert optimal_lineup["PG"]._name == "Player A"
    assert optimal_lineup["SG"]._name == "Player B"

    # Verify negative-projection player is excluded
    assert "C1" not in optimal_lineup

    # Verify only two players were selected
    assert len(optimal_lineup) == 2

    # Verify the optimal projected total
    assert lineup_total(optimal_lineup) == pytest.approx(105.0)
