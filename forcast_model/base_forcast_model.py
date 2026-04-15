from abc import ABC, abstractmethod
import torch

from core.config import Config


class BaseForecastModel(ABC):
    @classmethod
    @abstractmethod
    def search_space(cls):
        pass

    def fit(self, X_train, Y_train):
        pass

    @abstractmethod
    def evaluate(self, X_test, Y_test) -> float:
        pass

    @abstractmethod
    def train_step(self, X: torch.Tensor, Y: torch.Tensor, criterion: torch.nn.Module) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def eval_mode(self):
        pass

    @abstractmethod
    def train_mode(self):
        pass

    def freeze(self):
        pass

    def unfreeze(self):
        pass
