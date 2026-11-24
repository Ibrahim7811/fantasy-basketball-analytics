import pandas as pd
import math
from fantasy_basketball.model_persistence import load_sklearn_model, load_neural_network
from fantasy_basketball.torch_model import predict_neural_network
import sqlite3
from fantasy_basketball.database import load_player_game_logs
from fantasy_basketball.data_processing import prepare_game_logs
from fantasy_basketball.player_class import Player
from fantasy_basketball.lineup import build_optimal_lineup, lineup_total

def build_next_game_features(prepared_df: pd.DataFrame, game_date: str, feature_columns: list[str]) -> pd.DataFrame:
    if prepared_df.empty:
        raise ValueError("Game history cannot be empty")
    if not feature_columns:
        raise ValueError("At least one feature is required")
    supported_features = {
        "FP_AVG_5",
        "FP_AVG_10",
        "REST_DAYS",
        "MIN_AVG_5",
        "MIN_AVG_10"}
    if any(column not in supported_features for column in feature_columns):
        raise ValueError("Unknown feature column")
    history = prepared_df.copy()
    history["GAME_DATE"] = pd.to_datetime(history["GAME_DATE"], errors="raise")
    if history["GAME_DATE"].isna().any():
        raise ValueError("Game history contains missing dates")
    history = history.sort_values("GAME_DATE")
    upcoming_date = pd.Timestamp(game_date)
    if pd.isna(upcoming_date):
        raise ValueError("Upcoming game date cannot be missing")
    upcoming_date = upcoming_date.normalize()
    latest_date = history["GAME_DATE"].iloc[-1].normalize()
    if upcoming_date <= latest_date:
        raise ValueError("Upcoming game date must be after the latest completed game")
    features = {
        "FP_AVG_5": float(history["FANTASY_POINTS"].tail(5).mean()),
        "FP_AVG_10": float(history["FANTASY_POINTS"].tail(10).mean()),
        "REST_DAYS": (upcoming_date - latest_date).days - 1,
        "MIN_AVG_5": float(history["MIN"].tail(5).mean()),
        "MIN_AVG_10": float(history["MIN"].tail(10).mean())}
    return pd.DataFrame([features], columns=feature_columns)

def predict_next_game_sklearn(prepared_df: pd.DataFrame, game_date: str, model_path: str) -> float:
    bundle = load_sklearn_model(model_path)
    features = build_next_game_features(prepared_df,game_date,bundle["feature_columns"])
    predictions = bundle["model"].predict(features)
    projection = float(predictions[0])
    if not math.isfinite(projection):
        raise ValueError("Prediction must be finite")
    return projection

def predict_next_game_neural_network(prepared_df: pd.DataFrame, game_date: str, model_path: str) -> float:
    bundle = load_neural_network(model_path)
    features = build_next_game_features(prepared_df,game_date,bundle["feature_columns"])
    predictions = predict_neural_network(bundle["model"],features,bundle["scaler"])
    projection = float(predictions[0])
    if not math.isfinite(projection):
        raise ValueError("Prediction must be finite")
    return projection

def predict_stored_player_next_game(
    connection: sqlite3.Connection,
    player_id: int,
    season: str,
    game_date: str,
    model_path: str,
    model_type: str = "sklearn",
    as_of_date: str | None = None,
    scheduled_dates: list[str] | None = None
) -> float:
    if model_type != "sklearn" and model_type != "pytorch":
        raise ValueError("Invalid model type")
    if model_type == "sklearn":
        bundle = load_sklearn_model(model_path)
    else:
        bundle = load_neural_network(model_path)
    logs = load_player_game_logs(connection, player_id, season)
    upcoming_date = pd.Timestamp(game_date)
    if pd.isna(upcoming_date):
        raise ValueError("Upcoming game date cannot be missing")
    upcoming_date = upcoming_date.normalize()
    history_cutoff = upcoming_date
    if as_of_date is not None:
        history_cutoff = pd.Timestamp(as_of_date)
        if pd.isna(history_cutoff):
            raise ValueError("History cutoff date cannot be missing")
        history_cutoff = history_cutoff.normalize()
        if history_cutoff > upcoming_date:
            raise ValueError("History cutoff cannot be after prediction date")
    if logs.empty:
        raise ValueError("No completed games before prediction date")
    logs = logs.copy()
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"], errors="raise")
    if logs["GAME_DATE"].isna().any():
        raise ValueError("Game history contains missing dates")
    logs = logs.loc[logs["GAME_DATE"].dt.normalize() < history_cutoff].copy()
    if logs.empty:
        raise ValueError("No completed games before prediction date")
    prepared = prepare_game_logs(logs,bundle["metadata"]["scoring_weights"])
    features = build_next_game_features(prepared,upcoming_date.strftime("%Y-%m-%d"),bundle["feature_columns"])
    if scheduled_dates is not None and "REST_DAYS" in features.columns:
        last_completed_date = logs["GAME_DATE"].max().strftime("%Y-%m-%d")
        features.loc[0, "REST_DAYS"] = calculate_projected_rest_days(
            last_completed_date,
            upcoming_date.strftime("%Y-%m-%d"),
            scheduled_dates,
            history_cutoff.strftime("%Y-%m-%d"))
    if model_type == "sklearn":
        predictions = bundle["model"].predict(features)
    else:
        predictions = predict_neural_network(bundle["model"],features,bundle["scaler"])
    projection = float(predictions[0])
    if not math.isfinite(projection):
        raise ValueError("Prediction must be finite")
    return projection

def project_roster_for_date(
    connection: sqlite3.Connection,
    players: list[Player],
    player_ids: dict[str, int],
    season: str,
    game_date: str,
    model_path: str,
    model_type: str = "sklearn",
    as_of_date: str | None = None
) -> list[Player]:
    projected_players = []
    for player in players:
        if not player.plays_on(game_date):
            continue
        if player._name not in player_ids:
            raise ValueError(f"Missing NBA player ID for {player._name}")
        player_id = player_ids[player._name]
        projection = predict_stored_player_next_game(
            connection,player_id,
            season,
            game_date,
            model_path,
            model_type,
            as_of_date=as_of_date,
            scheduled_dates=player._schedule)
        projected_players.append(
            Player(
                name=player._name,
                positions=player._positions.copy(),
                projected_points=projection,
                schedule=[game_date],
                team=player._team))
    return projected_players

def optimize_projected_day(
    connection: sqlite3.Connection,
    players: list[Player],
    player_ids: dict[str, int],
    season: str,
    game_date: str,
    lineup_slots: list[str],
    model_path: str,
    model_type: str = "sklearn",
    as_of_date: str | None = None
) -> tuple[dict[str, Player], float]:
    projected_players = project_roster_for_date(connection,players,player_ids,season,game_date,model_path,model_type,as_of_date=as_of_date)
    lineup = build_optimal_lineup(projected_players,game_date,lineup_slots)
    projected_total = lineup_total(lineup)
    return lineup, projected_total

def calculate_projected_rest_days(last_completed_date: str, game_date: str, scheduled_dates: list[str], as_of_date: str,) -> int:
    last_completed = pd.Timestamp(last_completed_date)
    upcoming = pd.Timestamp(game_date)
    cutoff = pd.Timestamp(as_of_date)
    if pd.isna(last_completed) or pd.isna(upcoming) or pd.isna(cutoff):
        raise ValueError("Rest-day dates cannot be missing")
    last_completed = last_completed.normalize()
    upcoming = upcoming.normalize()
    cutoff = cutoff.normalize()
    if not last_completed < cutoff <= upcoming:
        raise ValueError("Invalid rest-day date order")
    previous_game = last_completed
    for date_value in scheduled_dates:
        scheduled_date = pd.Timestamp(date_value)
        if pd.isna(scheduled_date):
            raise ValueError("Scheduled dates cannot be missing")
        scheduled_date = scheduled_date.normalize()
        if cutoff <= scheduled_date < upcoming:
            previous_game = max(previous_game, scheduled_date)
    return (upcoming - previous_game).days - 1