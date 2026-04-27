import numpy as np
import torch
import torch.nn as nn

from core.config import Config


class GeneratorModel(nn.Module):
    """
    Hybrid target generator.

    A smooth hourly linear backbone keeps Y realistic. A small neural residual
    reads feature/hour context and can add harder nonlinear patterns without
    taking over the whole data-generating process.
    """

    def __init__(self, num_features: int, hidden_dim: int = 32):
        super().__init__()
        self.config = Config()

        if num_features <= 0:
            raise ValueError("GeneratorModel requires at least one feature")

        self.num_features = num_features
        self.hidden_dim = hidden_dim

        b0, b = self.create_initial_b_params()
        self.b0 = nn.Parameter(torch.tensor(b0, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        self.register_buffer("initial_b0", self.b0.detach().clone())
        self.register_buffer("initial_b", self.b.detach().clone())

        residual_input_dim = num_features + 4
        self.residual_encoder = nn.Sequential(
            nn.Linear(residual_input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal_filter = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=1),
            nn.SiLU(),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.max_residual = 1.0
        self.feature_scale = float(np.sqrt(self.num_features))
        self._last_residual: torch.Tensor | None = None
        self._last_y: torch.Tensor | None = None

        self._init_residual_as_noop()

    def forward(
        self,
        hour_idx_or_x: int | torch.Tensor,
        x_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Supports both project call styles:
        - forward(hour_idx, x_features) -> scalar
        - forward(X_seq) where X_seq is [B, T, F] or [T, F] -> [B, T] or [T]
        """
        if x_features is not None:
            return self._forward_single(hour_idx_or_x, x_features)

        X_seq = hour_idx_or_x
        if not torch.is_tensor(X_seq):
            raise TypeError("GeneratorModel.forward expected a Tensor input")

        squeeze_batch = False
        if X_seq.dim() == 2:
            X_seq = X_seq.unsqueeze(0)
            squeeze_batch = True

        if X_seq.dim() != 3:
            raise ValueError("GeneratorModel.forward expects X_seq with shape [B, T, F]")
        if X_seq.shape[-1] != self.num_features:
            raise ValueError(
                f"Feature mismatch: expected {self.num_features}, got {X_seq.shape[-1]}"
            )

        _, seq_len, _ = X_seq.shape
        hours = torch.arange(seq_len, device=X_seq.device) % self.config.hours_per_week()
        X_model = self._standardize_sequence_features(X_seq)

        base = self._linear_backbone(X_model, hours)
        residual = self._neural_residual(X_model, hours)
        y = base + residual

        self._last_residual = residual
        self._last_y = y
        return y.squeeze(0) if squeeze_batch else y

    def _forward_single(
        self,
        hour_idx: int | torch.Tensor,
        x_features: torch.Tensor,
    ) -> torch.Tensor:
        hour = self._normalize_hour_idx(hour_idx)

        if x_features.shape[-1] != self.num_features:
            raise ValueError(
                f"Feature mismatch: expected {self.num_features}, got {x_features.shape[-1]}"
            )

        x_seq = x_features.view(1, 1, -1)
        hours = torch.tensor([hour], device=x_features.device)
        x_model = self._standardize_sequence_features(x_seq)
        y = self._linear_backbone(x_model, hours) + self._neural_residual(x_model, hours)
        return y.squeeze()

    def _normalize_hour_idx(self, hour_idx: int | torch.Tensor) -> int:
        if torch.is_tensor(hour_idx):
            hour_idx = int(hour_idx.detach().to("cpu").item())

        return int(hour_idx) % self.config.hours_per_week()

    def clamp_parameters(self):
        with torch.no_grad():
            self.b0.clamp_(
                self.config.generator_clamp_min,
                self.config.generator_clamp_max,
            )
            self.b.clamp_(
                self.config.generator_clamp_min,
                self.config.generator_clamp_max,
            )
            self.residual_scale.clamp_(0.0, self.max_residual)

    def regularization_loss(self) -> torch.Tensor:
        if self._last_residual is None:
            return self.b0.new_tensor(0.0)

        residual = self._last_residual
        magnitude = residual.pow(2).mean()
        centered = residual.mean().pow(2)

        if residual.shape[1] > 1:
            smoothness = (residual[:, 1:] - residual[:, :-1]).pow(2).mean()
        else:
            smoothness = residual.new_tensor(0.0)

        scale_penalty = self.residual_scale.pow(2)
        if self._last_y is not None and self._last_y.numel() > 1:
            y_std = self._last_y.std(unbiased=False)
            target_std = residual.new_tensor(self.config.generator_target_std)
            y_scale = torch.relu(y_std - target_std).pow(2) / target_std.clamp_min(1.0).pow(2)
            y_smoothness = self._target_roughness(self._last_y)
        else:
            y_scale = residual.new_tensor(0.0)
            y_smoothness = residual.new_tensor(0.0)

        drift = (self.b0 - self.initial_b0).pow(2).mean() + (self.b - self.initial_b).pow(2).mean()
        coeff_smoothness = self._coefficient_smoothness()

        return (
            magnitude
            + 0.5 * smoothness
            + 0.1 * centered
            + 0.01 * scale_penalty
            + self.config.generator_scale_weight * y_scale
            + self.config.generator_drift_weight * drift
            + self.config.generator_coeff_smoothness_weight * coeff_smoothness
            + self.config.generator_y_smoothness_weight * y_smoothness
        )

    def create_initial_b_params(self):
        hours = self.config.hours_per_week()

        b0 = self._smooth_weekly_values(
            hours=hours,
            low=self.config.init_b0_min,
            high=self.config.init_b0_max,
        )

        b = np.vstack(
            [
                self._smooth_weekly_values(
                    hours=hours,
                    low=self.config.init_b1_min,
                    high=self.config.init_b1_max,
                )
                for _ in range(self.num_features)
            ]
        )

        return b0, b

    def _linear_backbone(self, X_seq: torch.Tensor, hours: torch.Tensor) -> torch.Tensor:
        coeffs = self.b[:, hours].transpose(0, 1).to(X_seq.device)
        bias = self.b0[hours].to(X_seq.device)
        contribution = torch.einsum("btf,tf->bt", X_seq, coeffs)
        return bias.unsqueeze(0) + self.config.generator_backbone_gain * contribution / self.feature_scale

    def _coefficient_smoothness(self) -> torch.Tensor:
        b0_diff = torch.diff(self.b0, append=self.b0[:1])
        b_diff = torch.diff(self.b, dim=1, append=self.b[:, :1])
        return b0_diff.pow(2).mean() + b_diff.pow(2).mean()

    @staticmethod
    def _target_roughness(y: torch.Tensor) -> torch.Tensor:
        if y.shape[1] < 3:
            return y.new_tensor(0.0)

        second_diff = y[:, 2:] - 2.0 * y[:, 1:-1] + y[:, :-2]
        y_var = y.var(unbiased=False).clamp_min(1e-3)
        return second_diff.pow(2).mean() / y_var

    @staticmethod
    def _standardize_sequence_features(X_seq: torch.Tensor) -> torch.Tensor:
        if X_seq.shape[1] <= 1:
            return X_seq

        mean = X_seq.mean(dim=(0, 1), keepdim=True)
        std = X_seq.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(1e-3)
        return torch.clamp((X_seq - mean) / std, -5.0, 5.0)

    def _neural_residual(self, X_seq: torch.Tensor, hours: torch.Tensor) -> torch.Tensor:
        hour_context = self._hour_context(hours).unsqueeze(0).expand(X_seq.shape[0], -1, -1)
        residual_input = torch.cat([X_seq, hour_context], dim=-1)
        h = self.residual_encoder(residual_input)
        h = h + self.temporal_filter(h.transpose(1, 2)).transpose(1, 2)
        raw_residual = self.residual_head(h).squeeze(-1)
        bounded_residual = torch.tanh(raw_residual)
        return torch.clamp(self.residual_scale, 0.0, self.max_residual) * bounded_residual

    def _hour_context(self, hours: torch.Tensor) -> torch.Tensor:
        hours = hours.to(dtype=torch.float32)
        weekly_phase = 2.0 * torch.pi * hours / self.config.hours_per_week()
        daily_phase = 2.0 * torch.pi * (hours % self.config.hours_per_day) / self.config.hours_per_day
        return torch.stack(
            [
                torch.sin(weekly_phase),
                torch.cos(weekly_phase),
                torch.sin(daily_phase),
                torch.cos(daily_phase),
            ],
            dim=-1,
        )

    def _init_residual_as_noop(self):
        last_layer = self.residual_head[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    @staticmethod
    def _smooth_weekly_values(hours: int, low: float, high: float) -> np.ndarray:
        weekly_phase = np.linspace(0.0, 2.0 * np.pi, hours, endpoint=False)
        daily_phase = np.linspace(0.0, 14.0 * np.pi, hours, endpoint=False)

        center = np.random.uniform(low * 0.25, high * 0.25)
        weekly_amp = np.random.uniform(0.15, 0.45) * (high - low)
        daily_amp = np.random.uniform(0.03, 0.15) * (high - low)
        weekly_shift = np.random.uniform(0.0, 2.0 * np.pi)
        daily_shift = np.random.uniform(0.0, 2.0 * np.pi)

        values = (
            center
            + weekly_amp * np.sin(weekly_phase + weekly_shift)
            + daily_amp * np.sin(daily_phase + daily_shift)
        )

        return np.clip(values, low, high).astype(np.float32)
