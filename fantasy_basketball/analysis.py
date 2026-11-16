from fantasy_basketball.nba_data import  get_player_id, get_player_game_logs
from fantasy_basketball.data_processing import prepare_game_logs
from fantasy_basketball.projections import predict_next_game
from fantasy_basketball.evaluation import evaluate_baseline

def analyze_player(player_name: str, season: str, scoring_weights: dict[str, float],) -> dict[str, int | float]:
    player_id = get_player_id(player_name)
    game_logs = get_player_game_logs(player_id,season)
    if len(game_logs) < 2:
        raise ValueError("At least two games are required for analysis")
    df = prepare_game_logs(game_logs,scoring_weights)
    projection = predict_next_game(df)
    baseline = evaluate_baseline(df)
    return {"games_analyzed": len(df), "projected_fp": projection, "baseline_mae": baseline,}