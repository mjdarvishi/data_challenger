from dataclasses import dataclass

import torch
from typing import Optional


@dataclass
class DataPoint:
    global_time: int
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
    execution_time: float
    forecast_time: float
    generator_time: float
    model_losses: dict[int, float]
    generator_loss: dict[int, float]
    params: dict
    data: list[DataPoint]
    predictions: Optional[torch.Tensor]
    targets: torch.Tensor
