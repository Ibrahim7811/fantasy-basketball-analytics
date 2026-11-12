
import pytest

from player_class import Player
from weekly import optimize_week


@pytest.fixture
def sample_players():
    return [
        Player(
            "Player A", ["PG", "SG"], 60.0,
            ["2026-11-09", "2026-11-11"], "DEN"
        ),
        Player(
            "Player B", ["PG"], 59.0,
            ["2026-11-09"], "BOS"
        ),
        Player(
            "Player C", ["SG"], 55.0,
            ["2026-11-10", "2026-11-11"], "DET"
        )
    ]


def test_single_day_matchup(sample_players):
    results, total = optimize_week(
        sample_players,
        "2026-11-09",
        "2026-11-09",
        ["PG", "SG"]
    )

    assert len(results) == 1
    assert "2026-11-09" in results

    lineup = results["2026-11-09"]["lineup"]

    assert lineup["PG"]._name == "Player B"
    assert lineup["SG"]._name == "Player A"
    assert total == pytest.approx(119.0)


def test_seven_day_matchup(sample_players):
    results, total = optimize_week(
        sample_players,
        "2026-11-09",
        "2026-11-15",
        ["PG", "SG"]
    )

    assert len(results) == 7

    expected_dates = [
        "2026-11-09",
        "2026-11-10",
        "2026-11-11",
        "2026-11-12",
        "2026-11-13",
        "2026-11-14",
        "2026-11-15"
    ]

    assert list(results.keys()) == expected_dates


def test_weekly_projected_total(sample_players):
    results, total = optimize_week(
        sample_players,
        "2026-11-09",
        "2026-11-15",
        ["PG", "SG"]
    )

    assert results["2026-11-09"]["projected_points"] == pytest.approx(119.0)
    assert results["2026-11-10"]["projected_points"] == pytest.approx(55.0)
    assert results["2026-11-11"]["projected_points"] == pytest.approx(115.0)

    assert total == pytest.approx(289.0)

    daily_sum = sum(
        result["projected_points"]
        for result in results.values()
    )

    assert total == pytest.approx(daily_sum)


def test_reversed_date_range(sample_players):
    with pytest.raises(
        ValueError,
        match="Start date cannot be after end date"
    ):
        optimize_week(
            sample_players,
            "2026-11-15",
            "2026-11-09",
            ["PG", "SG"]
        )


def test_no_scheduled_games(sample_players):
    results, total = optimize_week(
        sample_players,
        "2026-11-16",
        "2026-11-18",
        ["PG", "SG"]
    )

    assert len(results) == 3
    assert all(
        result["lineup"] == {}
        for result in results.values()
    )
    assert all(
        result["projected_points"] == pytest.approx(0.0)
        for result in results.values()
    )
    assert total == pytest.approx(0.0)
