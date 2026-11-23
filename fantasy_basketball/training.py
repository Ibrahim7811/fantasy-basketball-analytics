from pathlib import Path
import pandas as pd
from fantasy_basketball.models import train_linear_regression, train_random_forest
from fantasy_basketball.torch_model import train_neural_network
from fantasy_basketball.model_persistence import save_sklearn_model, save_neural_network

def train_and_save_models(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    output_dir: str,
    feature_columns: list[str],
    metadata: dict[str, object],
) -> dict[str, str]:
    model = train_linear_regression(train_df,feature_columns=feature_columns)
    forest_model = train_random_forest(train_df,random_state=42,feature_columns=feature_columns)
    neural_network, scaler, _ = train_neural_network(
        train_df,
        validation_df,
        epochs=3000,
        learning_rate=0.001,
        patience=30,
        random_state=42,
        feature_columns=feature_columns)
    linear_path = Path(output_dir) / "linear_regression.pkl"
    forest_path = Path(output_dir) / "random_forest.pkl"
    neural_path = Path(output_dir) / "neural_network.pt"
    save_sklearn_model(model, str(linear_path), metadata)
    save_sklearn_model(forest_model, str(forest_path), metadata)
    save_neural_network(neural_network,scaler,str(neural_path),metadata)
    return {"linear_regression": str(linear_path),"random_forest": str(forest_path),"neural_network": str(neural_path)}