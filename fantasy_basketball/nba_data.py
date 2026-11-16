from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import requests
import pandas as pd

def get_player_id(player_name: str) -> int:
    player_list = players.find_players_by_full_name(player_name)
    if not player_list:
        raise ValueError("Player not found")
    player_id = player_list[0]['id']
    return player_id

def get_player_game_logs(player_id: int, season: str) -> pd.DataFrame:
    try:
        game_log = playergamelog.PlayerGameLog(player_id=player_id,season=season,timeout=10)
        df = game_log.player_game_log.get_data_frame()
        return df
    except requests.exceptions.Timeout:
       raise
    except requests.exceptions.ConnectionError as error:
        raise

if __name__ == "__main__":
    player_id = get_player_id("Nikola Jokic")
    df = get_player_game_logs(player_id, "2025-26")

    print(df[["GAME_DATE", "PTS", "REB", "AST", "STL", "BLK", "TOV"]].head())
    print(f"Total games: {len(df)}")
