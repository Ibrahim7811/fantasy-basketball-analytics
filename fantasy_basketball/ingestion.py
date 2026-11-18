import sqlite3
import re
from nba_api.stats.static import teams as nba_teams
from fantasy_basketball.database import upsert_team, upsert_game
from fantasy_basketball.nba_data import get_season_schedule
from fantasy_basketball.data_processing import normalize_schedule

def sync_teams(connection: sqlite3.Connection) -> dict[str, int]:
    team_lookup = {}
    for team in nba_teams.get_teams():
        team_id = team["id"]
        name = team["full_name"]
        abbreviation = team["abbreviation"]
        upsert_team(connection,team_id, name, abbreviation)
        team_lookup[abbreviation] = team_id
    return team_lookup

def parse_matchup(matchup: str, team_lookup: dict[str, int],) -> tuple[int, int, int]:
    parts = matchup.split()

    if len(parts) != 3:
        raise ValueError("Invalid matchup format")
    
    team_abbr, separator, opponent_abbr = parts

    if separator not in ("vs.", "@"):
        raise ValueError("Invalid matchup separator")
    if team_abbr not in team_lookup or opponent_abbr not in team_lookup:
        raise ValueError("Unknown team abbreviation")
    
    player_team_id = team_lookup[team_abbr]
    opponent_team_id = team_lookup[opponent_abbr]

    if player_team_id == opponent_team_id:
        raise ValueError("A team cannot play itself")
    if separator == "vs.":
        return (player_team_id, player_team_id, opponent_team_id)
    return (player_team_id, opponent_team_id, player_team_id)

def sync_schedule(connection: sqlite3.Connection, season: str, team_lookup: dict[str, int],) -> int:
    schedule_df = get_season_schedule(season)
    schedule_df = normalize_schedule(schedule_df, team_lookup)
    for row in schedule_df.itertuples(index=False):
        game_id=row.GAME_ID
        game_date=row.GAME_DATE.strftime("%Y-%m-%d")
        home_team=int(row.HOME_TEAM_ID)
        away_team=int(row.AWAY_TEAM_ID)
        upsert_game(connection, game_id, game_date, season, home_team, away_team)
    return len(schedule_df)