from dataclasses import dataclass
from enum import Enum


class XFeature(Enum):
    X1 = "x1"
    X2 = "x2"
    X3 = "x3"
    X4 = "x4"
    X5 = "x5"
    X6 = "x6"
    X7 = "x7"
    X8 = "x8"
    X9 = "x9"
    X10 = "x10"


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

    generator_learning_rate = 3e-1
    forcaster_learning_rate = 1e-4
    forcaster_trainer_learning_rate = 1e-4
    generator_clamp_min = -10.0
    generator_clamp_max = 10.0
    training_epochs = 10
    forcast_trainer_epoch = 10
    generator_epoch = 10
    grade_search_epochs = 10
    input_dim: int = 6

    @staticmethod
    def total_samples():
        return Config.hours_per_day * Config.days_per_year

    @staticmethod
    def hours_per_week():
        return Config.hours_per_day * Config.days_per_week
    
    @staticmethod
    def to_dict():
        return {
            "hours_per_day": Config.hours_per_day,
            "days_per_week": Config.days_per_week,
            "days_per_year": Config.days_per_year,
            "seq_len": Config.seq_len,
            "pred_len": Config.pred_len,
            "train_ratio": Config.train_ratio,
            "batch_size": Config.batch_size,
            "noise_dim": Config.noise_dim,
            "init_b0_min": Config.init_b0_min,
            "init_b0_max": Config.init_b0_max,
            "init_b1_min": Config.init_b1_min,
            "init_b1_max": Config.init_b1_max,
            "init_b2_min": Config.init_b2_min,
            "init_b2_max": Config.init_b2_max,
            "generator_learning_rate": Config.generator_learning_rate,
            "forcaster_learning_rate": Config.forcaster_learning_rate,
            "forcaster_trainer_learning_rate": Config.forcaster_trainer_learning_rate,
            "generator_clamp_min": Config.generator_clamp_min,
            "generator_clamp_max": Config.generator_clamp_max,
            "training_epochs": Config.training_epochs,
            "forcast_trainer_epoch": Config.forcast_trainer_epoch,
            "generator_epoch": Config.generator_epoch,
            "grade_search_epochs": Config.grade_search_epochs
        }