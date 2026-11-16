import pandas as pd

def predict_next_game(df: pd.DataFrame, window: int = 5,) -> float:
    if window < 1:
        raise ValueError("window must be at least 1")
    if df.empty:
        raise ValueError("Game logs cannot be empty")

    prediction_avg = float(df['FANTASY_POINTS'].tail(window).mean())
    return prediction_avg