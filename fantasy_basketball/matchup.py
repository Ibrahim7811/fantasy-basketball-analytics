from fantasy_basketball.player_class import Player
from fantasy_basketball.weekly import optimize_week


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