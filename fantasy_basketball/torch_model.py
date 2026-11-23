import torch
from torch import nn
import pandas as pd
import numpy as np
from copy import deepcopy
import math
from fantasy_basketball.training_data import FEATURE_COLUMNS
from sklearn.preprocessing import StandardScaler

class FantasyPointsNet(nn.Module):
    def __init__(self, input_size: int = len(FEATURE_COLUMNS)):
        super().__init__()
        if type(input_size) is not int or input_size < 1:
            raise ValueError("input_size must be a positive integer")
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1))
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)

def fit_feature_scaler(train_df: pd.DataFrame, feature_columns: list[str] | None = None) -> StandardScaler:
    if train_df.empty:
        raise ValueError("Training data cannot be empty")
    columns = FEATURE_COLUMNS if feature_columns is None else feature_columns
    if not columns:
        raise ValueError("At least one feature is required")
    scaler = StandardScaler()
    scaler.fit(train_df[columns])
    return scaler

def prepare_tensors(df: pd.DataFrame, scaler: StandardScaler) -> tuple[torch.Tensor, torch.Tensor]:
    if df.empty:
        raise ValueError("Dataset cannot be empty")
    scaled_features = scaler.transform(df[scaler.feature_names_in_])
    inputs = torch.tensor(scaled_features, dtype=torch.float32)
    targets = torch.tensor(df["FANTASY_POINTS"].to_numpy(dtype="float32"),dtype=torch.float32)
    return inputs, targets

def train_one_epoch(model: FantasyPointsNet, inputs: torch.Tensor, targets: torch.Tensor, optimizer: torch.optim.Optimizer,) -> float:
    if inputs.shape[0] == 0 or targets.shape[0] == 0:
        raise ValueError("Training tensors cannot be empty")
    model.train()
    optimizer.zero_grad()
    predictions = model(inputs)
    loss = nn.MSELoss()(predictions, targets)
    loss.backward()
    optimizer.step()
    return float(loss.item())

def evaluate_network_loss(model: FantasyPointsNet, inputs: torch.Tensor, targets: torch.Tensor,) -> float:
    if inputs.shape[0] == 0 or targets.shape[0] == 0:
        raise ValueError("Evaluation tensors cannot be empty")
    model.eval()
    with torch.no_grad():
        predictions = model(inputs)
        loss = nn.MSELoss()(predictions, targets)
    return float(loss.item())

def train_neural_network(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    epochs: int = 500,
    learning_rate: float = 0.001,
    patience: int = 30,
    random_state: int = 42,
    feature_columns: list[str] | None = None,
) -> tuple[FantasyPointsNet, StandardScaler, dict[str, list[float]]]:
    if (
        train_df.empty
        or validation_df.empty
        or type(epochs) is not int
        or epochs <= 0
        or type(patience) is not int
        or patience <= 0
        or learning_rate <= 0
        or not math.isfinite(learning_rate)
    ):
        raise ValueError("Invalid input data")
    scaler = fit_feature_scaler(train_df,feature_columns=feature_columns)
    inputs, targets = prepare_tensors(train_df, scaler)
    validation_inputs, validation_targets = prepare_tensors(validation_df, scaler)
    torch.manual_seed(random_state)
    model = FantasyPointsNet(input_size=len(scaler.feature_names_in_))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"train_loss": [], "validation_loss": []}
    best_loss = float("inf")
    best_weights = None
    epochs_without_improvement = 0
    for _ in range(epochs):
        train_loss = train_one_epoch(model, inputs, targets, optimizer)
        validation_loss = evaluate_network_loss(model, validation_inputs, validation_targets)
        if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
            raise ValueError("Non-finite loss encountered")
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            epochs_without_improvement = 0
            best_weights = deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    model.load_state_dict(best_weights)
    model.eval()
    return model, scaler, history

def predict_neural_network(model: FantasyPointsNet, df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    if df.empty:
        raise ValueError("Prediction data cannot be empty")
    scaled_features = scaler.transform(df[scaler.feature_names_in_])
    inputs = torch.tensor(scaled_features, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        predictions = model(inputs)
    return predictions.cpu().numpy()