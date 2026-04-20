import numpy as np
import torch
import torch.nn as nn

from core.config import Config


class GeneratorModel(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.config = Config()
        if num_features <= 0:
            raise ValueError("GeneratorModel requires at least one feature")
        self.num_features = num_features
        # =========================================================
        # INIT PARAMETERS (use controlled initialization)
        # =========================================================
        b0, b = self.create_initial_b_params()

        self.b0 = nn.Parameter(torch.tensor(b0, dtype=torch.float32))
        # Shape: [num_features, hours_per_week]
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))

    def forward(self, hour_idx: int, x_features: torch.Tensor) -> torch.Tensor:
        coeffs = self.b[:, hour_idx]
        return self.b0[hour_idx] + torch.dot(coeffs, x_features)

    def clamp_parameters(self):
        with torch.no_grad():
            self.b0.copy_(self.b0.clamp(self.config.generator_clamp_min, self.config.generator_clamp_max))
            self.b.copy_(self.b.clamp(self.config.generator_clamp_min, self.config.generator_clamp_max))

    def create_initial_b_params(self):
            hours = self.config.hours_per_week()
            b0 = np.random.uniform(self.config.init_b0_min, self.config.init_b0_max, hours)
            b = np.random.uniform(self.config.init_b1_min, self.config.init_b1_max, (self.num_features, hours))
            return b0, b