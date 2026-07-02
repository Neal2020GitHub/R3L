import torch
import torch.nn as nn


class Softplus2(nn.Module):
    """
    Softplus-squared activation/loss: (softplus(x))^2
    Uses torch.nn.Softplus for numerical stability.

    Args:
        beta (float): Slope parameter for Softplus. Larger -> closer to ReLU.
        threshold (float): Values above this behave linearly to avoid overflow.
    """
    def __init__(self, beta: float = 1.0, threshold: float = 20.0):
        super().__init__()
        self.softplus = nn.Softplus(beta=beta, threshold=threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sp = self.softplus(x)
        return sp * sp


class ReLU2(nn.Module):
    """
    Relu-squared activation/loss: (relu(x))^2
    Uses torch.nn.ReLU for numerical stability.
    """
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x) * self.relu(x)
