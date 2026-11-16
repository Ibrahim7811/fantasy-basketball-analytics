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

def prepare_game_logs(df: pd.DataFrame, scoring_weights: dict[str, float], windows: tuple[int, ...] = (5, 10),) -> pd.DataFrame:
    prepped_df = add_rest_days(add_rolling_averages(calculate_fantasy_points(clean_game_logs(df),scoring_weights),windows))
    return prepped_df