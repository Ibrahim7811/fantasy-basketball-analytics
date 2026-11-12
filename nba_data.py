from nba_api.stats.static import players

def get_player_id(player_name: str) -> int:
    player_list = players.find_players_by_full_name(player_name)
    if not player_list:
        raise ValueError("Player not found")
    player_id = player_list[0]['id']
    return player_id

if __name__ == "__main__":
    print(get_player_id("Nikola Jokic"))