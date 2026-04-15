import torch
from core.config import Config


class TimeSeriesSplitter:
    def __init__(self):
        self.config = Config()
        self.train_ratio = self.config.train_ratio

    def split(self, X: torch.Tensor, Y: torch.Tensor):
        n = X.shape[0]
        split_idx = int(n * self.train_ratio)

        X_train = X[:split_idx]
        Y_train = Y[:split_idx]

        X_test = X[split_idx:]
        Y_test = Y[split_idx:]

        return X_train, Y_train, X_test, Y_test
