import numpy as np
import torch
import torch.nn as nn

from core.config import Config


class GeneratorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        # =========================================================
        # INIT PARAMETERS (use controlled initialization)
        # =========================================================
        b0, b1, b2 = self.create_initial_b_params()

        self.b0 = nn.Parameter(torch.tensor(b0, dtype=torch.float32))
        self.b1 = nn.Parameter(torch.tensor(b1, dtype=torch.float32))
        self.b2 = nn.Parameter(torch.tensor(b2, dtype=torch.float32))

    def forward(self, hour_idx:int, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.b0[hour_idx] + self.b1[hour_idx] * x1 + self.b2[hour_idx] * x2

    def clamp_parameters(self):
        with torch.no_grad():
            self.b0.copy_(self.b0.clamp(self.config.generator_clamp_min, self.config.generator_clamp_max))
            self.b1.copy_(self.b1.clamp(self.config.generator_clamp_min, self.config.generator_clamp_max))
            self.b2.copy_(self.b2.clamp(self.config.generator_clamp_min, self.config.generator_clamp_max))

    def create_initial_b_params(self):
            b0 = np.random.uniform(self.config.init_b0_min, self.config.init_b0_max, self.config.hours_per_week())
            b1 = np.random.uniform(self.config.init_b1_min, self.config.init_b1_max, self.config.hours_per_week())
            b2 = np.random.uniform(self.config.init_b2_min, self.config.init_b2_max, self.config.hours_per_week())
            return b0, b1, b2