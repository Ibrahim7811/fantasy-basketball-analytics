import pickle
from pathlib import Path
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from fantasy_basketball.torch_model import FantasyPointsNet

def save_sklearn_model(model: LinearRegression | RandomForestRegressor, file_path: str, metadata: dict[str, object]) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model,"feature_columns": model.feature_names_in_.tolist(),"metadata": metadata}
    with open(path,'wb') as f:
        pickle.dump(bundle,f,protocol=pickle.HIGHEST_PROTOCOL)

def load_sklearn_model(file_path: str) -> dict[str, object]:
    with open(file_path,'rb') as f:
        bundle = pickle.load(f)
        return bundle

def save_neural_network(model: FantasyPointsNet, scaler: StandardScaler, file_path: str, metadata: dict[str, object]) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_state_dict": model.state_dict(),
        "input_size": model.network[0].in_features,
        "scaler": scaler,
        "feature_columns": scaler.feature_names_in_.tolist(),
        "metadata": metadata}
    torch.save(bundle, path)

def load_neural_network(file_path: str) -> dict[str, object]:
    bundle = torch.load(file_path,map_location="cpu",weights_only=False)
    model = FantasyPointsNet(input_size=bundle["input_size"])
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return {"model": model,"scaler": bundle["scaler"],"feature_columns": bundle["feature_columns"],"metadata": bundle["metadata"]}