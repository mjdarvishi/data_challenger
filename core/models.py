from dataclasses import dataclass

import torch


@dataclass
class DataPoint:
    hour_of_week: int
    x1: float
    x2: float
    b0_used: float
    b1_used: float
    b2_used: float
    y: float


@dataclass
class StepRecord:
    step: int
    model_loss: float
    generator_loss: float
    params: dict
    data: list[DataPoint]
    predictions: torch.Tensor 
    targets: torch.Tensor     