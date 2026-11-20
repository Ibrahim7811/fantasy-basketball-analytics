import pytest
import torch
from fantasy_basketball.torch_model import FantasyPointsNet
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