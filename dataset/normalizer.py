import torch


class DataNormalizer:
    def __init__(self):
        self.X_mean: torch.Tensor = None
        self.X_std: torch.Tensor = None
        self.Y_mean: torch.Tensor = None
        self.Y_std: torch.Tensor = None

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        self.X_mean = X.mean(0, keepdim=True)
        self.X_std = X.std(0, keepdim=True) + 1e-6

        self.Y_mean = Y.mean()
        self.Y_std = Y.std() + 1e-6

    def transform(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        X_norm = (X - self.X_mean) / self.X_std
        Y_norm = (Y - self.Y_mean) / self.Y_std
        return X_norm, Y_norm

    def inverse_Y(self, Y_norm: torch.Tensor) -> torch.Tensor:
        return Y_norm * self.Y_std + self.Y_mean
