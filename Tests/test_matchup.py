import pytest
from fantasy_basketball.player_class import Player
from fantasy_basketball.matchup import analyze_matchup


@pytest.mark.parametrize(
    "team_a_points, team_b_points, expected_winner, expected_diff",
    [
        (150.0, 120.0, "Team A", 30.0),
        (100.0, 180.0, "Team B", -80.0),
        (150.0, 150.0, "Tie", 0.0)
    ]
)
def test_matchup_outcomes(
    team_a_points,
    team_b_points,
    expected_winner,
    expected_diff
):
    date = "2026-11-09"
    lineup_slots = ["PG"]

    team_a = [
        Player("Player A", ["PG"], team_a_points, [date], "DEN")
    ]

    team_b = [
        Player("Player B", ["PG"], team_b_points, [date], "BOS")
    ]

    results = analyze_matchup(
        team_a, team_b, date, date, lineup_slots
    )

    # Verify projected winner
    assert results["projected_winner"] == expected_winner

    # Verify signed point differential
    assert results["point_diff"] == pytest.approx(expected_diff)

    # Verify both teams' projected totals
    assert results["team_a"]["projected_total"] == pytest.approx(
        team_a_points
    )
    assert results["team_b"]["projected_total"] == pytest.approx(
        team_b_points
    )

    # Verify weekly results were preserved
    assert date in results["team_a"]["weekly_results"]
    assert date in results["team_b"]["weekly_results"]

    # Verify daily lineup assignments
    team_a_lineup = results["team_a"]["weekly_results"][date]["lineup"]
    team_b_lineup = results["team_b"]["weekly_results"][date]["lineup"]

    assert team_a_lineup["PG"]._name == "Player A"
    assert team_b_lineup["PG"]._name == "Player B"

def test_multi_day_matchup():
    team_a = [
        Player(
            "Player A1", ["PG"], 50.0,
            ["2026-11-09", "2026-11-11"], "DEN"
        ),
        Player(
            "Player A2", ["SG"], 40.0,
            ["2026-11-09", "2026-11-10"], "BOS"
        ),
        Player(
            "Player A3", ["PG", "SG"], 30.0,
            ["2026-11-10", "2026-11-11"], "DET"
        )
    ]

    team_b = [
        Player(
            "Player B1", ["PG"], 60.0,
            ["2026-11-09", "2026-11-11"], "LAL"
        ),
        Player(
            "Player B2", ["SG"], 35.0,
            ["2026-11-10"], "CLE"
        ),
        Player(
            "Player B3", ["PG", "SG"], 25.0,
            ["2026-11-09", "2026-11-10"], "NYK"
        )
    ]

    start_date = "2026-11-09"
    end_date = "2026-11-11"
    lineup_slots = ["PG", "SG"]

    results = analyze_matchup(
        team_a, team_b,
        start_date, end_date, lineup_slots
    )

    team_a_results = results["team_a"]["weekly_results"]
    team_b_results = results["team_b"]["weekly_results"]

    # Verify three days were analyzed
    assert len(team_a_results) == 3
    assert len(team_b_results) == 3

    # Verify Team A's daily projections
    assert team_a_results["2026-11-09"]["projected_points"] == pytest.approx(90.0)
    assert team_a_results["2026-11-10"]["projected_points"] == pytest.approx(70.0)
    assert team_a_results["2026-11-11"]["projected_points"] == pytest.approx(80.0)

    # Verify Team B's daily projections
    assert team_b_results["2026-11-09"]["projected_points"] == pytest.approx(85.0)
    assert team_b_results["2026-11-10"]["projected_points"] == pytest.approx(60.0)
    assert team_b_results["2026-11-11"]["projected_points"] == pytest.approx(60.0)

    # Verify weekly totals
    assert results["team_a"]["projected_total"] == pytest.approx(240.0)
    assert results["team_b"]["projected_total"] == pytest.approx(205.0)

    # Verify winner and point differential
    assert results["point_diff"] == pytest.approx(35.0)
    assert results["projected_winner"] == "Team A"

    # Verify no duplicate players and valid assignments
    for daily_results in (team_a_results, team_b_results):
        for day, data in daily_results.items():
            lineup = data["lineup"]

            assert len(lineup.values()) == len(set(lineup.values()))

            for slot, player in lineup.items():
                assert player.eligible_for(slot)
                assert player.plays_on(day)
