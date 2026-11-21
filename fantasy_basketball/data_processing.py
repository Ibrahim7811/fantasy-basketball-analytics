import pandas as pd

def clean_game_logs(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()
    clean_df['GAME_DATE'] = pd.to_datetime(clean_df["GAME_DATE"], format="%b %d, %Y")
    clean_df = clean_df.sort_values("GAME_DATE", ascending=True).reset_index(drop=True)
    return clean_df

def calculate_fantasy_points(df: pd.DataFrame, scoring_weights: dict[str, float],) -> pd.DataFrame:
    points_df = df.copy()
    points_df['FANTASY_POINTS'] = 0.0
    for stat, weight in scoring_weights.items():
        points_df["FANTASY_POINTS"] += points_df[stat] * weight
    return points_df

def add_rolling_averages(df: pd.DataFrame, windows: tuple[int, ...] = (5, 10),) -> pd.DataFrame:
    rolling_df = df.copy()
    previous_scores = rolling_df["FANTASY_POINTS"].shift(1)

    for window in windows:
        rolling_df[f"FP_AVG_{window}"] = (previous_scores.rolling(window=window, min_periods=1).mean())
    return rolling_df

def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    rest_df = df.copy()

    rest_df["REST_DAYS"] = (rest_df["GAME_DATE"].diff().dt.days - 1)
    return rest_df

def prepare_game_logs(df: pd.DataFrame, scoring_weights: dict[str, float], windows: tuple[int, ...] = (5, 10)) -> pd.DataFrame:
    prepped_df = clean_game_logs(df)
    prepped_df = calculate_fantasy_points(prepped_df, scoring_weights)
    prepped_df = add_rolling_averages(prepped_df, windows)
    prepped_df = add_minutes_features(prepped_df, windows)
    return add_rest_days(prepped_df)

def normalize_schedule(df: pd.DataFrame,team_lookup: dict[str, int],) -> pd.DataFrame:
    column_mapping = {
        "gameId": "GAME_ID",
        "gameDate": "GAME_DATE",
        "homeTeam_teamId": "HOME_TEAM_ID",
        "awayTeam_teamId": "AWAY_TEAM_ID",}

    if df.empty:
        empty_df = pd.DataFrame(columns=list(column_mapping.values()))
        empty_df["GAME_DATE"] = pd.to_datetime(empty_df["GAME_DATE"])
        return empty_df

    schedule_df = df[list(column_mapping)].copy()
    schedule_df = schedule_df.rename(columns=column_mapping)
    valid_games = (
        schedule_df["GAME_ID"].str.startswith("002", na=False)
        & schedule_df["HOME_TEAM_ID"].isin(team_lookup.values())
        & schedule_df["AWAY_TEAM_ID"].isin(team_lookup.values()))
    schedule_df = schedule_df.loc[valid_games].copy()
    schedule_df["GAME_DATE"] = pd.to_datetime(schedule_df["GAME_DATE"],format="%m/%d/%Y %H:%M:%S",)
    return (schedule_df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True))

def add_minutes_features(df: pd.DataFrame, windows: tuple[int, ...] = (5, 10)) -> pd.DataFrame:
    minutes_df = df.copy()
    previous_minutes = minutes_df["MIN"].shift(1)
    for window in windows:
        minutes_df[f"MIN_AVG_{window}"] = (previous_minutes.rolling(window=window, min_periods=1).mean())
    return minutes_df