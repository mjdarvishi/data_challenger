import torch
import torch.nn as nn

from core.config import Config


class GeneratorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.b0 = nn.Parameter(torch.randn(self.config.hours_per_week))
        self.b1 = nn.Parameter(torch.randn(self.config.hours_per_week))
        self.b2 = nn.Parameter(torch.randn(self.config.hours_per_week))

    def forward(self, hour_idx, x1, x2) -> torch.Tensor:
        return self.b0[hour_idx] + self.b1[hour_idx] * x1 + self.b2[hour_idx] * x2


    def compute_loss(self, X, Y):
        preds = self.forward(X[:,0], X[:,1], X[:,2])
        return torch.nn.functional.mse_loss(preds, Y)