import numpy as np
import torch
import torch.nn as nn

from core.config import Config


import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorModel(nn.Module):
    """
    Stable adversarial generator:
    - learns hard perturbations
    - keeps distribution stable
    - avoids chaotic regime switching
    """

    def __init__(self, num_features, hidden_dim=64):
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # =========================================
        # 1. Feature encoder (structure learning)
        # =========================================
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # =========================================
        # 2. Temporal memory (stability)
        # =========================================
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # =========================================
        # 3. Base generator (realistic signal)
        # =========================================
        self.base_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # =========================================
        # 4. Adversarial residual generator
        # =========================================
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # =========================================
        # 5. Difficulty controller (VERY IMPORTANT)
        # =========================================
        self.difficulty = nn.Parameter(torch.tensor(0.1))  # learnable scalar

    def forward(self, X_seq):
        """
        X_seq: [B, T, F]
        """

        # 1. Encode
        h = self.encoder(X_seq)  # [B, T, H]

        # 2. Temporal memory
        h, _ = self.rnn(h)

        # 3. Base signal (stable realistic structure)
        base = self.base_head(h)

        # 4. Adversarial perturbation (bounded)
        delta = self.delta_head(h)

        # IMPORTANT: bound perturbation
        delta = torch.tanh(delta)  # [-1, 1]

        # 5. Difficulty scaling (smooth curriculum)
        difficulty = torch.sigmoid(self.difficulty)

        # FINAL OUTPUT
        y = base + difficulty * delta

        return y.squeeze(-1)