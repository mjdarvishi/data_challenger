from enum import Enum

from utils import cal_input_dimenion


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
    X11 = "x11"
    X12 = "x12"
    X13 = "x13"


class SplitMode(Enum):
    CHRONOLOGICAL = "CHRONOLOGICAL"
    WEEKLY_BLOCK = "WEEKLY_BLOCK"


class Config:
    
    debug: bool = True
    
    hours_per_day: int = 24
    days_per_week: int = 7
    days_per_year: int = 365
    seq_len: int = 24
    pred_len: int = 6
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    split_mode: SplitMode = SplitMode.CHRONOLOGICAL
    split_seed: int = 42
    batch_size: int = 32
    noise_dim: int = 16
    init_b0_min = -5.0
    init_b0_max = 5.0
    init_b1_min = -3.0
    init_b1_max = 3.0
    init_b2_min = -3.0
    init_b2_max = 3.0

    generator_trainer_learning_rate = 3e-1
    forcaster_trainer_learning_rate = 1e-4
    generator_clamp_min = -10.0
    generator_clamp_max = 10.0
    training_epochs = 10
    forcast_trainer_epoch = 10
    generator_epoch = 10
    grade_search_epochs = 10
    input_dim: int = None  # to be set in main based on selected features
    
    @staticmethod
    def set_input_dim( features: list[XFeature]):
        Config.input_dim = cal_input_dimenion(features)


    noise_std: float = 2
    lag_gamma: float = 0.5
    
    alpha: float = 0.6
    beta: float = 0.3
    
    spike_prob: float = 0.02
    spike_amplitude: float = 10.0
    structural_phi_amp: float = 0.8
    structural_gamma: float = 0.2
    structural_alpha: float = 0.15
    structural_delta: float = 0.3

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
            "val_ratio": Config.val_ratio,
            "split_mode": Config.split_mode.value,
            "split_seed": Config.split_seed,
            "batch_size": Config.batch_size,
            "noise_dim": Config.noise_dim,
            "init_b0_min": Config.init_b0_min,
            "init_b0_max": Config.init_b0_max,
            "init_b1_min": Config.init_b1_min,
            "init_b1_max": Config.init_b1_max,
            "init_b2_min": Config.init_b2_min,
            "init_b2_max": Config.init_b2_max,
            "generator_trainer_learning_rate": Config.generator_trainer_learning_rate,
            "forcaster_trainer_learning_rate": Config.forcaster_trainer_learning_rate,
            "generator_clamp_min": Config.generator_clamp_min,
            "generator_clamp_max": Config.generator_clamp_max,
            "training_epochs": Config.training_epochs,
            "forcast_trainer_epoch": Config.forcast_trainer_epoch,
            "generator_epoch": Config.generator_epoch,
            "grade_search_epochs": Config.grade_search_epochs,
            "input_dim": Config.input_dim,
            "noise_std": Config.noise_std,
            "lag_gamma": Config.lag_gamma,
            "alpha": Config.alpha,
            "beta": Config.beta,
            "spike_prob": Config.spike_prob,
            "spike_amplitude": Config.spike_amplitude,
            "structural_phi_amp": Config.structural_phi_amp,
            "structural_gamma": Config.structural_gamma,
            "structural_alpha": Config.structural_alpha,
            "structural_delta": Config.structural_delta,
        }