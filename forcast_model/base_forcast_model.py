from abc import ABC, abstractmethod
import torch

from core.config import Config


class BaseForecastModel(ABC):
    
    @classmethod
    @abstractmethod
    def search_space(cls)-> dict:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def eval_mode(self):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def train_mode(self):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def freeze(self):
        raise NotImplementedError("Subclasses must implement this method")
    @abstractmethod
    def unfreeze(self):
        raise NotImplementedError("Subclasses must implement this method")
    @property
    @abstractmethod
    def parameters(self):
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abstractmethod
    def named_parameters(self):
        raise NotImplementedError("Subclasses must implement this method")