import numpy as np
import torch

from core.config import Config


class XFeatureGenerator:
    def __init__(self, name: str):
        self.name = name

    def generate(self, t: int) -> float:
        raise NotImplementedError

    def generate_numpy(self, t_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def generate_torch(self, t_tensor: torch.Tensor) -> torch.Tensor:
        values = self.generate_numpy(t_tensor.detach().cpu().numpy())
        return torch.tensor(values, device=t_tensor.device, dtype=torch.float)


class YearlySineGenerator(XFeatureGenerator):
    """
    X1 Generator

    Simple yearly periodic signal.

    Formula
    -------
    X1(t) = sin(2πt / P_year)

    where:
        t       = time index (hours)
        P_year  = yearly period (typically 8760 hours)
    """

    def __init__(self, name: str = "X1"):
        super().__init__(name)
        self.period = Config.total_samples()

    def generate(self, t: int) -> float:
        return np.sin(2 * np.pi * t / self.period)

    def generate_numpy(self, t_values: np.ndarray) -> np.ndarray:
        return np.sin(2 * np.pi * t_values / self.period)

    def generate_torch(self, t_tensor: torch.Tensor) -> torch.Tensor:
        return torch.sin(2 * torch.pi * t_tensor / self.period)


class ConstantGenerator(XFeatureGenerator):
    """
    X2 Generator

    Constant signal.

    Formula
    -------
    X2(t) = c

    where:
        c = constant value
    """

    def __init__(self, name: str = "X2", value: float = 1.0):
        super().__init__(name)
        self.value = value

    def generate(self, t: int) -> float:
        return self.value

    def generate_numpy(self, t_values: np.ndarray) -> np.ndarray:
        return np.full(shape=t_values.shape[0], fill_value=self.value)

    def generate_torch(self, t_tensor: torch.Tensor) -> torch.Tensor:
        return torch.full_like(t_tensor, fill_value=self.value, dtype=torch.float)


class YearlyWeeklySineNoiseGenerator(XFeatureGenerator):
    """
    X3 Generator

    Combined yearly and weekly periodic signal with additive noise.

    Formula
    -------
    X3(t) =
        sin(2πt / P_year)
        + sin(2πt / P_week)
        + ε_t

    where:
        t       = time index
        P_year  = yearly period (8760 hours)
        P_week  = weekly period (168 hours)
        ε_t     = Gaussian noise
    """

    def __init__(self, name: str = "X3", noise_std: float = 0.05):
        super().__init__(name)
        self.yearly_period = Config.total_samples()
        self.weekly_period = Config.hours_per_week()
        self.noise_std = noise_std

    def generate(self, t: int) -> float:
        s1 = np.sin(2 * np.pi * t / self.yearly_period)
        s2 = np.sin(2 * np.pi * t / self.weekly_period)
        noise = np.random.normal(loc=0.0, scale=self.noise_std)
        return s1 + s2 + noise

    def generate_numpy(self, t_values: np.ndarray) -> np.ndarray:
        s1 = np.sin(2 * np.pi * t_values / self.yearly_period)
        s2 = np.sin(2 * np.pi * t_values / self.weekly_period)
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=t_values.shape[0])
        return s1 + s2 + noise

    def generate_torch(self, t_tensor: torch.Tensor) -> torch.Tensor:
        s1 = torch.sin(2 * torch.pi * t_tensor / self.yearly_period)
        s2 = torch.sin(2 * torch.pi * t_tensor / self.weekly_period)
        noise = torch.randn_like(t_tensor, dtype=torch.float) * self.noise_std
        return s1 + s2 + noise


class ConstantWithNoiseGenerator(XFeatureGenerator):
    """
    X4: Constant with noise
    value + noise
    """

    def __init__(self, name: str = "X4", value: float = 1.0, noise_std: float = 0.05):
        super().__init__(name)
        self.value = value
        self.noise_std = noise_std

    def generate(self, t: int) -> float:
        noise = np.random.normal(loc=0.0, scale=self.noise_std)
        return self.value + noise

    def generate_numpy(self, t_values: np.ndarray) -> np.ndarray:
        base = np.full(shape=t_values.shape[0], fill_value=self.value)
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=t_values.shape[0])
        return base + noise

    def generate_torch(self, t_tensor: torch.Tensor) -> torch.Tensor:
        base = torch.full_like(t_tensor, fill_value=self.value, dtype=torch.float)
        noise = torch.randn_like(t_tensor, dtype=torch.float) * self.noise_std
        return base + noise


class TemperatureSeasonalGenerator(XFeatureGenerator):
    """
    X5: Realistic temperature signal
    yearly seasonality + daily cycle + noise
    Formula:
    -------
    X5(t) =
        A_year * sin(2πt / 8760)
        + A_day * sin(2πt / 24)
        + ε_t

    where:
        t        = time index (hours)
        8760     = hours per year
        24       = hours per day
        A_year   = yearly amplitude
        A_day    = daily amplitude
        ε_t      = Gaussian noise
    """

    def __init__(
        self,
        name: str = "X5",
        yearly_amp: float = 10.0,
        daily_amp: float = 4.0,
        noise_std: float = 0.5,
    ):
        super().__init__(name)
        self.yearly_period = Config.total_samples()
        self.daily_period = Config.hours_per_day
        self.yearly_amp = yearly_amp
        self.daily_amp = daily_amp
        self.noise_std = noise_std

    def generate(self, t: int) -> float:
        yearly = self.yearly_amp * np.sin(2 * np.pi * t / self.yearly_period)

        daily = self.daily_amp * np.sin(2 * np.pi * t / self.daily_period)

        noise = np.random.normal(
            0,
            self.noise_std,
        )

        return yearly + daily + noise

    def generate_numpy(self, t_values: np.ndarray) -> np.ndarray:

        yearly = self.yearly_amp * np.sin(2 * np.pi * t_values / self.yearly_period)

        daily = self.daily_amp * np.sin(2 * np.pi * t_values / self.daily_period)

        noise = np.random.normal(
            0,
            self.noise_std,
            size=t_values.shape[0],
        )

        return yearly + daily + noise

    def generate_torch(self, t_tensor: torch.Tensor) -> torch.Tensor:

        yearly = self.yearly_amp * torch.sin(
            2 * torch.pi * t_tensor / self.yearly_period
        )

        daily = self.daily_amp * torch.sin(2 * torch.pi * t_tensor / self.daily_period)

        noise = torch.randn_like(t_tensor, dtype=torch.float) * self.noise_std

        return yearly + daily + noise


class TemperatureStructuralGenerator(XFeatureGenerator):
    """
    X6 Generator

    Structural temperature model with seasonal cycles and thermal inertia.

    Formula
    -------
    X6(t) =
        A_year * sin(2πt / 8760)
        + A_day * sin(2πt / 24)
        + γ * X6(t-1)
        + ε_t

    where:
        t        = time index
        γ        = persistence coefficient (thermal inertia)
        X6(t-1)  = previous temperature value
        ε_t      = Gaussian noise
    """

    def __init__(
        self,
        name: str = "X6",
        yearly_amp: float = 10.0,
        daily_amp: float = 4.0,
        gamma: float = 0.9,
        noise_std: float = 0.3,
    ):
        super().__init__(name)

        self.yearly_period = Config.total_samples()
        self.daily_period = Config.hours_per_day
        self.yearly_amp = yearly_amp
        self.daily_amp = daily_amp
        self.gamma = gamma
        self.noise_std = noise_std

    def generate(self, t: int) -> float:
        yearly = self.yearly_amp * np.sin(2 * np.pi * t / self.yearly_period)

        daily = self.daily_amp * np.sin(2 * np.pi * t / self.daily_period)

        noise = np.random.normal(
            0,
            self.noise_std,
        )

        lag = self.gamma * self.generate(t - 1) if t > 0 else 0

        return yearly + daily + lag + noise

    def generate_numpy(self, t_values: np.ndarray) -> np.ndarray:

        T = np.zeros(len(t_values))

        yearly = self.yearly_amp * np.sin(2 * np.pi * t_values / self.yearly_period)

        daily = self.daily_amp * np.sin(2 * np.pi * t_values / self.daily_period)

        noise = np.random.normal(
            0,
            self.noise_std,
            len(t_values),
        )

        for t in range(len(t_values)):
            lag = self.gamma * T[t - 1] if t > 0 else 0

            T[t] = yearly[t] + daily[t] + lag + noise[t]

        return T

    def generate_torch(self, t_tensor: torch.Tensor) -> torch.Tensor:

        values = self.generate_numpy(t_tensor.detach().cpu().numpy())

        return torch.tensor(
            values,
            device=t_tensor.device,
            dtype=torch.float,
        )
