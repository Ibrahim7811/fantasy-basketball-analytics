from tabulate import tabulate

def format_matchup_report(results: dict) -> str:
    team_a_weekly_total = results['team_a']['projected_total']
    team_b_weekly_total = results['team_b']['projected_total']
    projected_winner = results['projected_winner']
    point_diff = results['point_diff']
    team_a_daily = results["team_a"]["weekly_results"]
    team_b_daily = results["team_b"]["weekly_results"]
    table_data = []

    for day in team_a_daily:
        team_a_projected_points = team_a_daily[day]['projected_points']
        team_b_projected_points = team_b_daily[day]['projected_points']
        table_data.append([day,team_a_projected_points,team_b_projected_points])
    formatted_table = tabulate(table_data,headers=["Date", "Team A", "Team B"],tablefmt="grid",floatfmt=".2f")
    report = (
    "\nHEAD-TO-HEAD MATCHUP REPORT\n\n"
    f"{formatted_table}\n\n"
    f"Weekly Total - Team A: {team_a_weekly_total:.2f}\n"
    f"Weekly Total - Team B: {team_b_weekly_total:.2f}\n\n"
    f"Projected Winner: {projected_winner}\n"
    f"Point Differential: {point_diff:+.2f}")
    return report