
from report import format_matchup_report
import pytest


def test_format_matchup_report():
    results = {
        "team_a": {
            "weekly_results": {
                "2026-11-09": {
                    "lineup": {},
                    "projected_points": 90.0
                },
                "2026-11-10": {
                    "lineup": {},
                    "projected_points": 70.0
                }
            },
            "projected_total": 160.0
        },
        "team_b": {
            "weekly_results": {
                "2026-11-09": {
                    "lineup": {},
                    "projected_points": 85.0
                },
                "2026-11-10": {
                    "lineup": {},
                    "projected_points": 60.0
                }
            },
            "projected_total": 145.0
        },
        "point_diff": 15.0,
        "projected_winner": "Team A"
    }

    report = format_matchup_report(results)

    # Verify the function returns a string
    assert isinstance(report, str)

    # Verify the report title
    assert "HEAD-TO-HEAD MATCHUP REPORT" in report

    # Verify both dates appear
    assert "2026-11-09" in report
    assert "2026-11-10" in report

    # Verify daily projections appear in the table
    assert "90.00" in report
    assert "70.00" in report
    assert "85.00" in report
    assert "60.00" in report

    # Verify weekly totals
    assert "Weekly Total - Team A: 160.00" in report
    assert "Weekly Total - Team B: 145.00" in report

    # Verify the projected winner and differential
    assert "Projected Winner: Team A" in report
    assert "Point Differential: +15.00" in report

@pytest.mark.parametrize(
    "team_a_total, team_b_total, winner, diff",
    [
        (150.0, 120.0, "Team A", 30.0),
        (100.0, 180.0, "Team B", -80.0),
        (150.0, 150.0, "Tie", 0.0)
    ]
)
def test_report_matchup_outcomes(
    team_a_total,
    team_b_total,
    winner,
    diff
):
    date = "2026-11-09"

    results = {
        "team_a": {
            "weekly_results": {
                date: {
                    "lineup": {},
                    "projected_points": team_a_total
                }
            },
            "projected_total": team_a_total
        },
        "team_b": {
            "weekly_results": {
                date: {
                    "lineup": {},
                    "projected_points": team_b_total
                }
            },
            "projected_total": team_b_total
        },
        "point_diff": diff,
        "projected_winner": winner
    }

    report = format_matchup_report(results)

    assert f"Projected Winner: {winner}" in report
    assert f"Point Differential: {diff:+.2f}" in report
    assert f"Weekly Total - Team A: {team_a_total:.2f}" in report
    assert f"Weekly Total - Team B: {team_b_total:.2f}" in report
