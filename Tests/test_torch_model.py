import pytest
import torch
import numpy as np
import pandas as pd
from fantasy_basketball import torch_model
from fantasy_basketball.torch_model import FantasyPointsNet, fit_feature_scaler, prepare_tensors, train_one_epoch, evaluate_network_loss, train_neural_network, predict_neural_network
from fantasy_basketball.training_data import FEATURE_COLUMNS

@pytest.mark.parametrize("batch_size", [1, 6])
def test_network_returns_one_finite_prediction_per_game(batch_size):
    model = FantasyPointsNet()
    inputs = torch.randn(batch_size, len(FEATURE_COLUMNS))
    predictions = model(inputs)
    assert predictions.shape == (batch_size,)
    assert torch.isfinite(predictions).all().item()

def test_network_supports_backpropagation():
    model = FantasyPointsNet()
    inputs = torch.randn(6, len(FEATURE_COLUMNS))
    targets = torch.full((6,), 10.0)
    predictions = model(inputs)
    loss = torch.nn.MSELoss()(predictions, targets)
    loss.backward()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert parameters
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all().item()
    assert any(parameter.grad.abs().sum().item() > 0 for parameter in parameters)

def test_network_preserves_inputs():
    model = FantasyPointsNet()
    inputs = torch.randn(6, len(FEATURE_COLUMNS))
    original = inputs.clone()
    model(inputs)
    torch.testing.assert_close(inputs, original, rtol=0, atol=0)

@pytest.fixture
def scaler_training_df():
    # Deliberately use a different column order and include extra columns.
    return pd.DataFrame({
            "PLAYER_ID": [10, 20],
            "REST_DAYS": [0.0, 2.0],
            "FANTASY_POINTS": [100.0, 200.0],
            "FP_AVG_10": [20.0, 40.0],
            "FP_AVG_5": [10.0, 30.0],
            "MIN_AVG_5": [20.0, 40.0],
            "MIN_AVG_10": [22.0, 42.0],
        })

def test_fit_feature_scaler_learns_only_selected_features(scaler_training_df,):
    scaler = fit_feature_scaler(scaler_training_df)
    assert list(scaler.feature_names_in_) == FEATURE_COLUMNS
    np.testing.assert_allclose(scaler.mean_, [20.0, 30.0, 1.0, 30.0, 32.0])
    np.testing.assert_allclose(scaler.scale_, [10.0, 10.0, 1.0, 10.0, 10.0])
    transformed = scaler.transform(scaler_training_df[FEATURE_COLUMNS])
    np.testing.assert_allclose(
        transformed,
        [
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])

def test_fitted_scaler_can_transform_future_data(scaler_training_df):
    scaler = fit_feature_scaler(scaler_training_df)
    original_means = scaler.mean_.copy()
    original_scales = scaler.scale_.copy()
    future_features = pd.DataFrame({
            "FP_AVG_5": [50.0],
            "FP_AVG_10": [60.0],
            "REST_DAYS": [4.0],
            "MIN_AVG_5": [60.0],
            "MIN_AVG_10": [62.0],
        })
    transformed = scaler.transform(future_features)
    np.testing.assert_allclose(transformed, [[3.0, 3.0, 3.0, 3.0, 3.0]])
    np.testing.assert_array_equal(scaler.mean_, original_means)
    np.testing.assert_array_equal(scaler.scale_, original_scales)

def test_fit_feature_scaler_rejects_empty_input(scaler_training_df):
    empty_df = scaler_training_df.iloc[:0].copy()
    with pytest.raises(ValueError, match="Training data cannot be empty"):
        fit_feature_scaler(empty_df)

def test_fit_feature_scaler_preserves_input(scaler_training_df):
    original = scaler_training_df.copy(deep=True)
    fit_feature_scaler(scaler_training_df)
    pd.testing.assert_frame_equal(scaler_training_df, original)

@pytest.mark.parametrize("row_count", [1, 2])
def test_prepare_tensors_values_shapes_and_row_alignment(scaler_training_df, row_count):
    scaler = fit_feature_scaler(scaler_training_df)
    # Reverse the rows to check that inputs and targets stay aligned.
    df = scaler_training_df.iloc[[1, 0]].iloc[:row_count].copy()
    inputs, targets = prepare_tensors(df, scaler)
    expected_inputs = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]],dtype=torch.float32)[:row_count]
    expected_targets = torch.tensor([200.0, 100.0],dtype=torch.float32)[:row_count]
    assert inputs.shape == (row_count, len(FEATURE_COLUMNS))
    assert targets.shape == (row_count,)
    assert inputs.dtype == torch.float32
    assert targets.dtype == torch.float32
    torch.testing.assert_close(inputs, expected_inputs)
    torch.testing.assert_close(targets, expected_targets)

def test_prepare_tensors_uses_existing_scaler_statistics(scaler_training_df,):
    scaler = fit_feature_scaler(scaler_training_df)
    original_means = scaler.mean_.copy()
    original_scales = scaler.scale_.copy()
    original_sample_count = scaler.n_samples_seen_
    future_df = pd.DataFrame({
            "FP_AVG_5": [50.0],
            "FP_AVG_10": [60.0],
            "REST_DAYS": [4.0],
            "MIN_AVG_5": [60.0],
            "MIN_AVG_10": [62.0],
            "FANTASY_POINTS": [75.0],
        })
    inputs, targets = prepare_tensors(future_df, scaler)
    torch.testing.assert_close(inputs,torch.tensor([[3.0, 3.0, 3.0, 3.0, 3.0]], dtype=torch.float32))
    torch.testing.assert_close(targets,torch.tensor([75.0], dtype=torch.float32))
    np.testing.assert_array_equal(scaler.mean_, original_means)
    np.testing.assert_array_equal(scaler.scale_, original_scales)
    assert scaler.n_samples_seen_ == original_sample_count

def test_prepare_tensors_rejects_empty_input(scaler_training_df):
    scaler = fit_feature_scaler(scaler_training_df)
    empty_df = scaler_training_df.iloc[:0].copy()
    with pytest.raises(ValueError, match="Dataset cannot be empty"):
        prepare_tensors(empty_df, scaler)

def test_prepare_tensors_preserves_input_and_returns_copies(scaler_training_df,):
    scaler = fit_feature_scaler(scaler_training_df)
    original = scaler_training_df.copy(deep=True)
    inputs, targets = prepare_tensors(scaler_training_df, scaler)
    pd.testing.assert_frame_equal(scaler_training_df, original)
    inputs[0, 0] = -999.0
    targets[0] = -999.0
    pd.testing.assert_frame_equal(scaler_training_df, original)

@pytest.fixture
def training_step_data():
    model = FantasyPointsNet()
    # Controlled parameters make the expected results predictable.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    inputs = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]],dtype=torch.float32)
    targets = torch.tensor([2.0, 4.0], dtype=torch.float32)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, inputs, targets, optimizer

def test_train_one_epoch_returns_loss_and_updates_model(training_step_data):
    model, inputs, targets, optimizer = training_step_data
    original_inputs = inputs.clone()
    original_targets = targets.clone()
    model.eval()
    loss = train_one_epoch(model, inputs, targets, optimizer)
    assert model.training
    assert isinstance(loss, float)
    # Initial predictions are zero: (2² + 4²) / 2 = 10.
    assert loss == pytest.approx(10.0)
    with torch.no_grad():
        predictions = model(inputs)
        updated_loss = torch.nn.MSELoss()(predictions, targets).item()
    torch.testing.assert_close(predictions,torch.tensor([0.6, 0.6], dtype=torch.float32))
    assert updated_loss < loss
    torch.testing.assert_close(inputs, original_inputs, rtol=0, atol=0)
    torch.testing.assert_close(targets, original_targets, rtol=0, atol=0)

def test_train_one_epoch_clears_previous_gradients(training_step_data):
    model, inputs, targets, optimizer = training_step_data
    # These artificial gradients must be cleared before training.
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 99.0)
    first_loss = train_one_epoch(model, inputs, targets, optimizer)
    second_loss = train_one_epoch(model, inputs, targets, optimizer)
    assert first_loss == pytest.approx(10.0)
    assert second_loss == pytest.approx(6.76)
    with torch.no_grad():
        predictions = model(inputs)
    torch.testing.assert_close(predictions,torch.tensor([1.08, 1.08], dtype=torch.float32))

@pytest.mark.parametrize(
    "empty_inputs, empty_targets",
    [(True, True), (True, False), (False, True)]
)
def test_train_one_epoch_rejects_empty_tensors(training_step_data, empty_inputs, empty_targets):
    model, inputs, targets, optimizer = training_step_data
    if empty_inputs:
        inputs = inputs[:0]
    if empty_targets:
        targets = targets[:0]
    with pytest.raises(ValueError, match="Training tensors cannot be empty"):
        train_one_epoch(model, inputs, targets, optimizer)

@pytest.mark.parametrize(
    "row_count, expected_loss",
    [(1, 4.0), (2, 10.0)]
)
def test_evaluate_network_loss_returns_mse(training_step_data, row_count, expected_loss):
    model, inputs, targets, _ = training_step_data
    model.train()
    loss = evaluate_network_loss(model, inputs[:row_count], targets[:row_count])
    assert isinstance(loss, float)
    assert loss == pytest.approx(expected_loss)
    assert not model.training

def test_evaluate_network_loss_disables_gradient_tracking(training_step_data):
    model, inputs, targets, _ = training_step_data
    inputs.requires_grad_()
    observed = []
    def record_forward(_module, _arguments, output):
        observed.append((torch.is_grad_enabled(), output.requires_grad))
    handle = model.register_forward_hook(record_forward)
    try:
        evaluate_network_loss(model, inputs, targets)
    finally:
        handle.remove()
    assert observed == [(False, False)]

def test_evaluate_network_loss_preserves_parameters_and_data(training_step_data):
    model, inputs, targets, _ = training_step_data
    original_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    original_inputs = inputs.clone()
    original_targets = targets.clone()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    evaluate_network_loss(model, inputs, targets)
    for parameter, original in zip(model.parameters(), original_parameters):
        torch.testing.assert_close(parameter.detach(), original, rtol=0, atol=0)
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad,torch.ones_like(parameter),rtol=0,atol=0)
    torch.testing.assert_close(inputs, original_inputs, rtol=0, atol=0)
    torch.testing.assert_close(targets, original_targets, rtol=0, atol=0)

@pytest.mark.parametrize(
    "empty_inputs, empty_targets",
    [(True, True), (True, False), (False, True)]
)
def test_evaluate_network_loss_rejects_empty_tensors(training_step_data, empty_inputs, empty_targets):
    model, inputs, targets, _ = training_step_data
    if empty_inputs:
        inputs = inputs[:0]
    if empty_targets:
        targets = targets[:0]
    with pytest.raises(ValueError, match="Evaluation tensors cannot be empty"):
        evaluate_network_loss(model, inputs, targets)

@pytest.fixture
def neural_training_frames(scaler_training_df):
    train_df = scaler_training_df.copy(deep=True)

    validation_df = pd.DataFrame({
            "FP_AVG_5": [50.0, 70.0],
            "FP_AVG_10": [60.0, 80.0],
            "REST_DAYS": [4.0, 6.0],
            "MIN_AVG_5": [60.0, 80.0],
            "MIN_AVG_10": [62.0, 82.0],
            "FANTASY_POINTS": [75.0, 95.0],
        })
    return train_df, validation_df

def test_train_neural_network_real_training(neural_training_frames):
    train_df, validation_df = neural_training_frames
    original_train = train_df.copy(deep=True)
    original_validation = validation_df.copy(deep=True)
    model, scaler, history = train_neural_network(train_df,validation_df,epochs=5,patience=10,random_state=7)
    assert isinstance(model, FantasyPointsNet)
    assert not model.training
    assert set(history) == {"train_loss", "validation_loss"}
    for losses in history.values():
        assert len(losses) == 5
        assert np.isfinite(losses).all()
    # Statistics must come exclusively from training features.
    np.testing.assert_allclose(scaler.mean_, [20.0, 30.0, 1.0, 30.0, 32.0])
    assert scaler.n_samples_seen_ == len(train_df)
    validation_inputs, validation_targets = prepare_tensors(validation_df, scaler)
    restored_loss = evaluate_network_loss(model, validation_inputs, validation_targets)
    assert restored_loss == pytest.approx(min(history["validation_loss"]))
    pd.testing.assert_frame_equal(train_df, original_train)
    pd.testing.assert_frame_equal(validation_df, original_validation)

@pytest.mark.parametrize(
    "losses, epochs, patience, expected_epochs, best_epoch",
    [
        # Equal losses count as no improvement.
        ([5.0, 3.0, 3.0, 4.0, 9.0], 10, 2, 4, 2),
        # A new best loss resets the patience counter.
        ([5.0, 4.0, 4.5, 3.0, 3.5, 4.0, 9.0], 10, 2, 6, 4),
        # Training respects the maximum epoch count.
        ([5.0, 4.0, 3.0], 3, 2, 3, 3),
    ],
)
def test_train_neural_network_stopping_and_best_weights(
    neural_training_frames,
    monkeypatch,
    losses,
    epochs,
    patience,
    expected_epochs,
    best_epoch,
):
    train_df, validation_df = neural_training_frames
    validation_losses = iter(losses)
    epoch_count = 0

    def fake_train(model, inputs, targets, optimizer):
        nonlocal epoch_count
        epoch_count += 1
        torch.testing.assert_close(inputs,torch.tensor([[-1.0, -1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0, 1.0]]))
        torch.testing.assert_close(targets, torch.tensor([100.0, 200.0]))
        # Mutate a real parameter so restoration can be checked.
        with torch.no_grad():
            model.network[-1].bias.fill_(float(epoch_count))
        return float(epoch_count)
    def fake_evaluate(model, inputs, targets):
        # Check that validation receives its own scaled data.
        torch.testing.assert_close(
            inputs,
            torch.tensor(
                [[3.0, 3.0, 3.0, 3.0, 3.0], [5.0, 5.0, 5.0, 5.0, 5.0]]
            ))
        torch.testing.assert_close(targets, torch.tensor([75.0, 95.0]))
        return next(validation_losses)
    monkeypatch.setattr(torch_model, "train_one_epoch", fake_train)
    monkeypatch.setattr(torch_model, "evaluate_network_loss", fake_evaluate)
    model, _, history = train_neural_network(train_df,validation_df,epochs=epochs,patience=patience)
    assert epoch_count == expected_epochs
    assert history["train_loss"] == [float(i) for i in range(1, expected_epochs + 1)]
    assert history["validation_loss"] == losses[:expected_epochs]
    assert model.network[-1].bias.item() == pytest.approx(best_epoch)
    assert not model.training

def test_train_neural_network_is_reproducible(neural_training_frames):
    train_df, validation_df = neural_training_frames
    first_model, _, first_history = train_neural_network(train_df, validation_df, epochs=3, random_state=7)
    second_model, _, second_history = train_neural_network(train_df, validation_df, epochs=3, random_state=7)
    for key in first_history:
        assert first_history[key] == pytest.approx(second_history[key])
    for first, second in zip(first_model.parameters(), second_model.parameters()):
        torch.testing.assert_close(first, second, rtol=0, atol=0)

@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"epochs": -1},
        {"epochs": 1.5},
        {"epochs": True},
        {"patience": 0},
        {"patience": -1},
        {"patience": 1.5},
        {"patience": True},
        {"learning_rate": 0},
        {"learning_rate": -0.1},
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"learning_rate": float("-inf")},
    ],
)
def test_train_neural_network_rejects_invalid_settings(neural_training_frames, kwargs):
    train_df, validation_df = neural_training_frames
    with pytest.raises(ValueError, match="Invalid input data"):
        train_neural_network(train_df, validation_df, **kwargs)

@pytest.mark.parametrize("empty_frame", ["training", "validation"])
def test_train_neural_network_rejects_empty_data(neural_training_frames, empty_frame):
    train_df, validation_df = neural_training_frames
    if empty_frame == "training":
        train_df = train_df.iloc[:0].copy()
    else:
        validation_df = validation_df.iloc[:0].copy()
    with pytest.raises(ValueError, match="Invalid input data"):
        train_neural_network(train_df, validation_df)

@pytest.mark.parametrize("bad_loss", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("bad_stage", ["training", "validation"])
def test_train_neural_network_rejects_nonfinite_loss(neural_training_frames, monkeypatch, bad_loss, bad_stage):
    train_df, validation_df = neural_training_frames
    monkeypatch.setattr(torch_model,"train_one_epoch",lambda *args: bad_loss if bad_stage == "training" else 1.0)
    monkeypatch.setattr(torch_model,"evaluate_network_loss",lambda *args: bad_loss if bad_stage == "validation" else 1.0)
    with pytest.raises(ValueError, match="Non-finite loss encountered"):
        train_neural_network(train_df, validation_df, epochs=1)

@pytest.fixture
def prediction_setup(scaler_training_df):
    scaler = fit_feature_scaler(scaler_training_df)
    model = FantasyPointsNet()
    # Use a simple layer with known weights to verify predictions.
    model.network = torch.nn.Linear(len(FEATURE_COLUMNS), 1)
    with torch.no_grad():
        model.network.weight.copy_(torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]]))
        model.network.bias.fill_(5.0)
    return model, scaler

@pytest.mark.parametrize("row_count", [1, 2])
def test_predict_neural_network_values_and_order(scaler_training_df, prediction_setup, row_count):
    model, scaler = prediction_setup
    # Targets are deliberately absent; reverse rows to check ordering.
    features = (scaler_training_df.drop(columns=["FANTASY_POINTS"]).iloc[[1, 0]].iloc[:row_count].copy())
    predictions = predict_neural_network(model, features, scaler)
    # Scaled rows contain five ones and five negative ones.
    # Prediction = 2*x1 + 3*x2 + 4*x3 + 5*x4 + 6*x5 + 5.
    expected = np.array([25.0, -15.0], dtype=np.float32)[:row_count]
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (row_count,)
    assert predictions.dtype == np.float32
    np.testing.assert_allclose(predictions, expected)

def test_predict_neural_network_preserves_data_model_and_scaler(prediction_setup):
    model, scaler = prediction_setup
    features = pd.DataFrame({
            "REST_DAYS": [4.0],
            "MIN_AVG_5": [60.0],
            "MIN_AVG_10": [62.0],
            "FP_AVG_10": [20.0],
            "FP_AVG_5": [50.0],
        },
        index=[99])
    original_features = features.copy(deep=True)
    original_means = scaler.mean_.copy()
    original_scales = scaler.scale_.copy()
    original_sample_count = scaler.n_samples_seen_
    original_weights = [parameter.detach().clone() for parameter in model.parameters()]
    predictions = predict_neural_network(model, features, scaler)
    # Scaled features: [3, -1, 3, 3, 3].
    # 2*3 + 3*(-1) + 4*3 + 5*3 + 6*3 + 5 = 53.
    np.testing.assert_allclose(predictions, [53.0])
    pd.testing.assert_frame_equal(features, original_features)
    np.testing.assert_array_equal(scaler.mean_, original_means)
    np.testing.assert_array_equal(scaler.scale_, original_scales)
    assert scaler.n_samples_seen_ == original_sample_count
    for parameter, original in zip(model.parameters(), original_weights):
        torch.testing.assert_close(parameter.detach(), original, rtol=0, atol=0)

def test_predict_neural_network_uses_evaluation_without_gradients(scaler_training_df, prediction_setup):
    model, scaler = prediction_setup
    features = scaler_training_df[FEATURE_COLUMNS]
    observed = []
    model.train()
    def record_forward(module, arguments, output):
        observed.append((module.training, torch.is_grad_enabled(), output.requires_grad))
    handle = model.register_forward_hook(record_forward)
    try:
        predict_neural_network(model, features, scaler)
    finally:
        handle.remove()
    assert observed == [(False, False, False)]

def test_predict_neural_network_rejects_empty_input(scaler_training_df, prediction_setup):
    model, scaler = prediction_setup
    empty_df = scaler_training_df[FEATURE_COLUMNS].iloc[:0].copy()
    with pytest.raises(ValueError, match="Prediction data cannot be empty"):
        predict_neural_network(model, empty_df, scaler)