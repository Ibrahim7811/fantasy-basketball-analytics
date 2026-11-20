import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from fantasy_basketball.training_data import FEATURE_COLUMNS

def train_linear_regression(train_df: pd.DataFrame,) -> LinearRegression:
    if train_df.empty:
        raise ValueError("Training data cannot be empty")
    
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["FANTASY_POINTS"]
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(train_df: pd.DataFrame, random_state: int = 42,) -> RandomForestRegressor:
    if train_df.empty:
        raise ValueError("Training data cannot be empty")

    inputs = train_df[FEATURE_COLUMNS]
    target = train_df['FANTASY_POINTS']
    model = RandomForestRegressor(n_estimators=100,min_samples_leaf=5,random_state=random_state,n_jobs=-1)
    model.fit(inputs,target)
    return model