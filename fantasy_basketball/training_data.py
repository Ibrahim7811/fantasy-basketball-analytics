import sqlite3
import pandas as pd
from fantasy_basketball.database import load_player_game_logs
from fantasy_basketball.data_processing import prepare_game_logs

FEATURE_COLUMNS = ["FP_AVG_5", "FP_AVG_10", "REST_DAYS"]

def build_training_dataset(connection: sqlite3.Connection, player_ids: list[int], seasons: list[str], scoring_weights: dict[str, float],) -> pd.DataFrame:
    frames = []
    for player_id in sorted(set(player_ids)):
        for season in sorted(set(seasons)):
            logs = load_player_game_logs(connection, player_id, season)
            if logs.empty:
                continue
            prepared = prepare_game_logs(logs, scoring_weights)
            prepared["SEASON"] = season
            frames.append(prepared)
    if not frames:
        raise ValueError("No game logs found for the requested players and seasons")

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.dropna(subset=FEATURE_COLUMNS + ["FANTASY_POINTS"])

    if dataset.empty:
        raise ValueError("No usable training rows remain")

    output_columns = [
        "GAME_ID",
        "PLAYER_ID",
        "TEAM_ID",
        "GAME_DATE",
        "SEASON",
        *FEATURE_COLUMNS,
        "FANTASY_POINTS"]
    return (dataset[output_columns].sort_values(["GAME_DATE", "PLAYER_ID", "GAME_ID"]).reset_index(drop=True))

def split_training_dataset(df: pd.DataFrame, test_fraction: float = 0.2,) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if df.empty:
        raise ValueError("Training dataset cannot be empty")
    
    sorted_df = df.sort_values(["GAME_DATE", "PLAYER_ID", "GAME_ID"])
    unique_dates = sorted_df["GAME_DATE"].drop_duplicates().tolist()
    if len(unique_dates) < 2:
        raise ValueError("At least two unique game dates are required")
    
    train_date_count = int(len(unique_dates) * (1 - test_fraction))
    train_date_count = max(1, min(train_date_count, len(unique_dates) - 1))
    cutoff_date = unique_dates[train_date_count]
    train_df = sorted_df.loc[sorted_df["GAME_DATE"] < cutoff_date].copy().reset_index(drop=True)
    test_df = sorted_df.loc[sorted_df["GAME_DATE"] >= cutoff_date].copy().reset_index(drop=True)
    return train_df, test_df