from datetime import date, timedelta
from player_class import Player
from lineup import build_optimal_lineup, lineup_total


def optimize_week(players: list[Player], start_date: str, end_date: str, lineup_slots: list[str]) -> tuple[dict, float]:
    start_date = date.fromisoformat(start_date)
    end_date = date.fromisoformat(end_date)
    if start_date > end_date:
        raise ValueError("Start date cannot be after end date")
    count = (end_date - start_date).days
    date_range = []
    weekly_results = {}
    for i in range(count + 1):
        date_range.append((start_date + timedelta(days=i)).isoformat())
    for day in date_range:
        daily_lineup = build_optimal_lineup(players,day,lineup_slots)
        weekly_results[day] = {'lineup': daily_lineup, 'projected_points': lineup_total(daily_lineup)}
    results = weekly_results.values()
    weekly_total = sum(result['projected_points'] for result in results)
    return weekly_results,weekly_total