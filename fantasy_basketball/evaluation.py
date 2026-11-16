import pandas as pd
from fantasy_basketball.projections import predict_next_game

def evaluate_baseline(df: pd.DataFrame, prediction_column: str = "FP_AVG_5",) -> float:
    cdf = df.dropna(subset=['FANTASY_POINTS', prediction_column])
    if cdf.empty:
        raise ValueError("No valid rows available for evaluation")
    error = float((cdf["FANTASY_POINTS"] - cdf[prediction_column]).abs().mean())
    return error