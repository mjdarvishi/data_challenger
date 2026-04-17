from dataclasses import dataclass
from enum import Enum


class XFeature(Enum):
    X1 = "x1"
    X2 = "x2"
    X3 = "x3"
    X4 = "x4"
    X5 = "x5"
    X6 = "x6"


class Config:
    hours_per_day: int = 24
    days_per_week: int = 7
    days_per_year: int = 365
    seq_len: int = 24
    pred_len: int = 6
    train_ratio: float = 0.8
    batch_size: int = 32
    noise_dim: int = 16
    init_b0_min = -5.0
    init_b0_max = 5.0
    init_b1_min = -3.0
    init_b1_max = 3.0
    init_b2_min = -3.0
    init_b2_max = 3.0

    generator_learning_rate = 2e-1
    forcaster_learning_rate = 1e-4
    forcaster_trainer_learning_rate = 1e-4
    generator_clamp_min = -10.0
    generator_clamp_max = 10.0
    training_epochs = 10
    forcast_trainer_epoch = 10
    generator_epoch = 10
    grade_search_epochs = 10

    total_samples = 8760
    hours_per_day = 24
    hours_per_week = 168
