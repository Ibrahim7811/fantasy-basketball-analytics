from unittest.mock import Mock
import pytest
from fantasy_basketball import inference as inference_module
from fantasy_basketball import matchup as matchup_module
from fantasy_basketball.matchup import analyze_matchup
from fantasy_basketball.player_class import Player


@pytest.mark.parametrize(
    "team_a_points, team_b_points, expected_winner, expected_diff",
    [
        (150.0, 120.0, "Team A", 30.0),
        (100.0, 180.0, "Team B", -80.0),
        (150.0, 150.0, "Tie", 0.0)
    ]
)
def test_matchup_outcomes(team_a_points,team_b_points,expected_winner,expected_diff):
    date = "2026-11-09"
    lineup_slots = ["PG"]
    team_a = [
        Player("Player A", ["PG"], team_a_points, [date], "DEN")]
    team_b = [
        Player("Player B", ["PG"], team_b_points, [date], "BOS")]
    results = analyze_matchup(team_a, team_b, date, date, lineup_slots)
    # Verify projected winner
    assert results["projected_winner"] == expected_winner
    # Verify signed point differential
    assert results["point_diff"] == pytest.approx(expected_diff)
    # Verify both teams' projected totals
    assert results["team_a"]["projected_total"] == pytest.approx(team_a_points)
    assert results["team_b"]["projected_total"] == pytest.approx(team_b_points)
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
            ["2026-11-10", "2026-11-11"], "DET")]
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
            ["2026-11-09", "2026-11-10"], "NYK")]
    start_date = "2026-11-09"
    end_date = "2026-11-11"
    lineup_slots = ["PG", "SG"]
    results = analyze_matchup(team_a,team_b,start_date,end_date,lineup_slots)
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

@pytest.mark.parametrize("model_type", ["sklearn", "pytorch"])
@pytest.mark.parametrize("points_a, points_b, winner, difference", [
    (40.0, 30.0, "Team A", 20.0),
    (25.0, 35.0, "Team B", -20.0),
    (30.0, 30.0, "Tie", 0.0)])
def test_analyze_projected_matchup_compares_weekly_totals(monkeypatch, model_type, points_a, points_b, winner, difference):
    schedule_a = ["2026-02-01", "2026-02-03"]
    schedule_b = ["2026-02-02", "2026-02-03"]
    team_a = [Player("Player A", ["PG"], 999.0, schedule_a.copy(), "BOS")]
    team_b = [Player("Player B", ["PG"], 888.0, schedule_b.copy(), "NYK")]
    player_ids = {"Player A": 101, "Player B": 202}
    projections = {101: points_a, 202: points_b}
    connection = object()
    cutoff = "2026-01-31"
    def predict_player(connection, player_id, season, game_date, model_path, model_type, as_of_date=None, scheduled_dates=None):
        return projections[player_id]
    predictor = Mock(side_effect=predict_player)
    monkeypatch.setattr(inference_module, "predict_stored_player_next_game", predictor)
    result = matchup_module.analyze_projected_matchup(
        connection,
        team_a,
        team_b,
        player_ids,
        "2025-26",
        "2026-02-01",
        "2026-02-03",
        ["PG"],
        "saved_model",
        model_type,
        as_of_date=cutoff)
    assert result["projected_winner"] == winner
    assert result["point_diff"] == pytest.approx(difference)
    assert result["team_a"]["projected_total"] == pytest.approx(points_a * 2)
    assert result["team_b"]["projected_total"] == pytest.approx(points_b * 2)
    expected_dates = ["2026-02-01", "2026-02-02", "2026-02-03"]
    for team_key in ("team_a", "team_b"):
        daily_results = result[team_key]["weekly_results"]
        assert list(daily_results) == expected_dates
        assert sum(day["projected_points"] for day in daily_results.values()) == pytest.approx(result[team_key]["projected_total"])
    assert result["team_a"]["weekly_results"]["2026-02-02"]["projected_points"] == 0.0
    assert result["team_b"]["weekly_results"]["2026-02-01"]["projected_points"] == 0.0
    assert result["team_a"]["weekly_results"]["2026-02-01"]["lineup"]["PG"]._name == "Player A"
    assert result["team_b"]["weekly_results"]["2026-02-02"]["lineup"]["PG"]._name == "Player B"
    assert predictor.call_count == 4
    for prediction_call in predictor.call_args_list:
        args = prediction_call.args
        kwargs = prediction_call.kwargs
        assert args[0] is connection
        assert args[2] == "2025-26"
        assert args[4] == "saved_model"
        assert args[5] == model_type
        assert kwargs["as_of_date"] == cutoff
        assert kwargs["scheduled_dates"] == (schedule_a if args[1] == 101 else schedule_b)
    assert team_a[0]._projected_points == 999.0
    assert team_b[0]._projected_points == 888.0
    assert team_a[0]._schedule == schedule_a
    assert team_b[0]._schedule == schedule_b

def test_analyze_projected_matchup_empty_teams_tie(monkeypatch):
    predictor = Mock()
    monkeypatch.setattr(inference_module, "predict_stored_player_next_game", predictor)
    result = matchup_module.analyze_projected_matchup(
        None,
        [],
        [],
        {},
        "2025-26",
        "2026-02-01",
        "2026-02-03",
        ["PG"],
        "unused.pkl")
    assert result["projected_winner"] == "Tie"
    assert result["point_diff"] == 0.0
    assert result["team_a"]["projected_total"] == 0.0
    assert result["team_b"]["projected_total"] == 0.0
    assert len(result["team_a"]["weekly_results"]) == 3
    assert len(result["team_b"]["weekly_results"]) == 3
    predictor.assert_not_called()

def test_analyze_projected_matchup_rejects_invalid_date_range():
    with pytest.raises(ValueError, match="Invalid date range"):
        matchup_module.analyze_projected_matchup(
            None,
            [],
            [],
            {},
            "2025-26",
            "2026-02-03",
            "2026-02-01",
            ["PG"],
            "unused.pkl")

def test_analyze_projected_matchup_propagates_prediction_failure(monkeypatch):
    player = Player("Player A", ["PG"], 20.0, ["2026-02-01"], "BOS")
    predictor = Mock(side_effect=ValueError("No completed games before prediction date"))
    monkeypatch.setattr(inference_module, "predict_stored_player_next_game", predictor)
    with pytest.raises(ValueError, match="No completed games"):
        matchup_module.analyze_projected_matchup(
            None,
            [player],
            [],
            {"Player A": 101},
            "2025-26",
            "2026-02-01",
            "2026-02-03",
            ["PG"],
            "unused.pkl")
    assert player._projected_points == 20.0