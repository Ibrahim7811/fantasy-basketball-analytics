from fantasy_basketball.player_class import Player
from fantasy_basketball.weekly import optimize_week
import sqlite3
from fantasy_basketball.inference import optimize_projected_week


def analyze_matchup(team_a: list[Player], team_b: list[Player], start_date: str, end_date: str, lineup_slots: list[str]) -> dict:
    team_a_weekly_results, team_a_weekly_total = optimize_week(team_a, start_date, end_date, lineup_slots)
    team_b_weekly_results, team_b_weekly_total = optimize_week(team_b, start_date, end_date, lineup_slots)
    point_diff = team_a_weekly_total - team_b_weekly_total
    if point_diff > 0:
        projected_winner = "Team A"
    elif point_diff < 0:
        projected_winner = "Team B"
    else:
        projected_winner = "Tie"
    results = {}
    results['team_a'] = {"weekly_results": team_a_weekly_results,"projected_total": team_a_weekly_total}
    results['team_b'] = {"weekly_results": team_b_weekly_results,"projected_total": team_b_weekly_total}
    results['projected_winner'] = projected_winner
    results['point_diff'] = point_diff
    return results

def analyze_projected_matchup(
    connection: sqlite3.Connection,
    team_a: list[Player],
    team_b: list[Player],
    player_ids: dict[str, int],
    season: str,
    start_date: str,
    end_date: str,
    lineup_slots: list[str],
    model_path: str,
    model_type: str = "sklearn",
    as_of_date: str | None = None,
) -> dict:
    team_a_weekly_results, team_a_weekly_total = optimize_projected_week(
        connection,
        team_a,
        player_ids,
        season,
        start_date,
        end_date,
        lineup_slots,
        model_path,
        model_type,
        as_of_date=as_of_date)
    team_b_weekly_results, team_b_weekly_total = optimize_projected_week(
        connection,
        team_b,
        player_ids,
        season,
        start_date,
        end_date,
        lineup_slots,
        model_path,
        model_type,
        as_of_date=as_of_date)
    point_diff = team_a_weekly_total - team_b_weekly_total
    if point_diff > 0:
        projected_winner = "Team A"
    elif point_diff < 0:
        projected_winner = "Team B"
    else:
        projected_winner = "Tie"
    return {
        "team_a": {
            "weekly_results": team_a_weekly_results,
            "projected_total": team_a_weekly_total},
        "team_b": {
            "weekly_results": team_b_weekly_results,
            "projected_total": team_b_weekly_total},
        "projected_winner": projected_winner,
        "point_diff": point_diff}