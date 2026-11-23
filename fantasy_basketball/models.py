import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from fantasy_basketball.training_data import FEATURE_COLUMNS

def train_linear_regression(train_df: pd.DataFrame, feature_columns: list[str] | None = None) -> LinearRegression:
    if train_df.empty:
        raise ValueError("Training data cannot be empty")
    columns = FEATURE_COLUMNS if feature_columns is None else feature_columns
    if not columns:
        raise ValueError("At least one feature is required")
    inputs = train_df[columns]
    target = train_df["FANTASY_POINTS"]
    model = LinearRegression()
    model.fit(inputs, target)
    return model

def train_random_forest(train_df: pd.DataFrame, random_state: int = 42, feature_columns: list[str] | None = None) -> RandomForestRegressor:
    if train_df.empty:
        raise ValueError("Training data cannot be empty")
    columns = FEATURE_COLUMNS if feature_columns is None else feature_columns
    if not columns:
        raise ValueError("At least one feature is required")
    inputs = train_df[columns]
    target = train_df["FANTASY_POINTS"]
    model = RandomForestRegressor(n_estimators=100,min_samples_leaf=5,random_state=random_state,n_jobs=-1)
    model.fit(inputs, target)
    return model