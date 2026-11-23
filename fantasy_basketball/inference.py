import pandas as pd
import math
from fantasy_basketball.model_persistence import load_sklearn_model, load_neural_network
from fantasy_basketball.torch_model import predict_neural_network
import sqlite3
from fantasy_basketball.database import load_player_game_logs
from fantasy_basketball.data_processing import prepare_game_logs

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
    model_type: str = "sklearn"
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
    if logs.empty:
        raise ValueError("No completed games before prediction date")
    logs = logs.copy()
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"], errors="raise")
    if logs["GAME_DATE"].isna().any():
        raise ValueError("Game history contains missing dates")
    logs = logs.loc[logs["GAME_DATE"].dt.normalize() < upcoming_date].copy()
    if logs.empty:
        raise ValueError("No completed games before prediction date")
    prepared = prepare_game_logs(logs,bundle["metadata"]["scoring_weights"])
    features = build_next_game_features(prepared,upcoming_date.strftime("%Y-%m-%d"),bundle["feature_columns"])
    if model_type == "sklearn":
        predictions = bundle["model"].predict(features)
    else:
        predictions = predict_neural_network(bundle["model"],features,bundle["scaler"])
    projection = float(predictions[0])
    if not math.isfinite(projection):
        raise ValueError("Prediction must be finite")
    return projection