import numpy as np
import torch
from typing import Dict

from core.config import Config


class XFeatureGenerator:
    def __init__(self, name: str):
        self.name = name

    def generate(self, t: int) -> float:
        raise NotImplementedError

    def generate_numpy(self, t_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def generate_numpy_with_context(
        self,
        t_values: np.ndarray,
        context: Dict[str, np.ndarray],
    ) -> np.ndarray:
        return self.generate_numpy(t_values)

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


class RegimeSwitchGenerator(XFeatureGenerator):
    """
    X7: Regime Switching Time Series

    ----------------------------
    Mathematical definition:
    ----------------------------

    X7(t) =
        f1(t)   if t < T1
        f2(t)   if T1 ≤ t < T2
        f3(t)   otherwise

    where:
        f1(t) = sin(2πt / P)
        f2(t) = a * t + b
        f3(t) = constant + noise

    ----------------------------
    Intuition:
    ----------------------------
    - Simulates structural breaks
    - Non-stationary process
    - Forces model to detect regime changes
    """

    def __init__(
        self,
        name: str = "X7",
        noise_std: float = 0.3,
    ):
        super().__init__(name)
        self.noise_std = noise_std
        self.P = Config.total_samples()

        self.T1 = int(self.P * 0.3)
        self.T2 = int(self.P * 0.6)

    def _f1(self, t):
        return np.sin(2 * np.pi * t / self.P)

    def _f2(self, t):
        return 0.01 * t + 0.5

    def _f3(self, t):
        return 2.0 + np.random.normal(0, self.noise_std)

    def generate(self, t: int) -> float:
        if t < self.T1:
            return self._f1(t)
        elif t < self.T2:
            return self._f2(t)
        else:
            return self._f3(t)

    def generate_numpy(self, t_values: np.ndarray) -> np.ndarray:
        out = np.zeros_like(t_values, dtype=float)

        for i, t in enumerate(t_values):
            out[i] = self.generate(int(t))

        return out

    def generate_torch(self, t_tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            self.generate_numpy(t_tensor.cpu().numpy()),
            device=t_tensor.device,
            dtype=torch.float,
        )
        
class DelayedDependencyGenerator(XFeatureGenerator):
    """
    X8: Temporal Memory Process (Lag-based)

    ----------------------------
    Mathematical definition:
    ----------------------------

    X8(t) =
        α * X8(t-1)
        + β * X8(t-24)
        + sin(2πt / P)
        + ε(t)

    where:
        α, β = decay factors
        ε(t) = noise

    ----------------------------
    Intuition:
    ----------------------------
    - Long + short term dependency
    - AR-like but nonlinear
    - Tests memory capacity of model
    """

    def __init__(
        self,
        name: str = "X8",
        alpha: float = 0.6,
        beta: float = 0.3,
        noise_std: float = 0.2,
    ):
        super().__init__(name)
        self.alpha = alpha
        self.beta = beta
        self.noise_std = noise_std
        self.P = Config.total_samples()
        self.cache = {}

    def generate(self, t: int) -> float:
        if t == 0:
            self.cache[t] = 0.0
            return 0.0

        lag1 = self.cache.get(t - 1, 0.0)
        lag24 = self.cache.get(t - 24, 0.0)

        value = (
            self.alpha * lag1
            + self.beta * lag24
            + np.sin(2 * np.pi * t / self.P)
            + np.random.normal(0, self.noise_std)
        )

        self.cache[t] = value
        return value

    def generate_numpy(self, t_values: np.ndarray) -> np.ndarray:
        self.cache = {}
        return np.array([self.generate(int(t)) for t in t_values])

    def generate_torch(self, t_tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            self.generate_numpy(t_tensor.cpu().numpy()),
            device=t_tensor.device,
            dtype=torch.float,
        )
        
class MultiplicativeInteractionGenerator(XFeatureGenerator):
    """
    X9: Nonlinear Feature Interaction

    ----------------------------
    Mathematical definition:
    ----------------------------

    Preferred dependency form (when x1..x4 are available):

        X9(t) = sin(X1(t)) * (1 + 0.5 * X2(t)) + cos(X3(t) + X4(t)) + ε(t)

    Fallback form (if dependencies are not present):

        X9(t) = sin(s1(t)) * (1 + 0.5 * s2(t)) + cos(s3(t) + s4(t)) + ε(t)

    ----------------------------
    Intuition:
    ----------------------------
    - Nonlinear coupling between signals
    - Breaks linear assumption of forecasting models
    - Forces interaction learning
    """

    def __init__(
        self,
        name: str = "X9",
        noise_std: float = 0.1,
    ):
        super().__init__(name)
        self.noise_std = noise_std
        self.p_day = Config.hours_per_day
        self.p_week = Config.hours_per_week()
        self.p_year = Config.total_samples()
        self.total = max(1, Config.total_samples() - 1)

    def _fallback_core_numpy(self, t_values: np.ndarray) -> np.ndarray:
        t = t_values.astype(float)
        t_norm = 2.0 * (t / self.total) - 1.0

        s1 = np.sin(2 * np.pi * t / self.p_day)
        s2 = np.cos(2 * np.pi * t / self.p_week)
        s3 = np.sin(2 * np.pi * t / self.p_year)
        s4 = np.tanh(2.0 * t_norm)

        return np.sin(s1) * (1.0 + 0.5 * s2) + np.cos(s3 + s4)

    def generate(self, t: int) -> float:
        base = self._fallback_core_numpy(np.array([t], dtype=float))[0]
        return float(base + np.random.normal(0, self.noise_std))

    def generate_numpy(self, t_values: np.ndarray) -> np.ndarray:
        base = self._fallback_core_numpy(t_values)
        noise = np.random.normal(0, self.noise_std, size=t_values.shape[0])
        return base + noise

    def generate_numpy_with_context(
        self,
        t_values: np.ndarray,
        context: Dict[str, np.ndarray],
    ) -> np.ndarray:
        x1 = context.get("x1")
        x2 = context.get("x2")
        x3 = context.get("x3")
        x4 = context.get("x4")

        if any(arr is None for arr in (x1, x2, x3, x4)):
            return self.generate_numpy(t_values)

        n = min(len(x1), len(x2), len(x3), len(x4), t_values.shape[0])
        base = np.sin(x1[:n]) * (1.0 + 0.5 * x2[:n]) + np.cos(x3[:n] + x4[:n])
        noise = np.random.normal(0, self.noise_std, size=n)
        out = np.zeros(t_values.shape[0], dtype=float)
        out[:n] = base + noise
        if n < t_values.shape[0]:
            out[n:] = self._fallback_core_numpy(t_values[n:])
        return out

    def generate_torch(self, t_tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            self.generate_numpy(t_tensor.cpu().numpy()),
            device=t_tensor.device,
            dtype=torch.float,
        )
        
class SparseSpikeGenerator(XFeatureGenerator):
    """
    X10: Rare Event Shock Process

    ----------------------------
    Mathematical definition:
    ----------------------------

    X10(t) =
        base(t) + S(t)

    where:
        S(t) =
            A  with probability p
            0  otherwise

    ----------------------------
    Intuition:
    ----------------------------
    - Rare high-magnitude spikes
    - Heavy-tailed behavior
    - Tests robustness of MSE models
    """

    def __init__(
        self,
        name: str = "X10",
        spike_prob: float = 0.02,
        spike_amplitude: float = 10.0,
    ):
        super().__init__(name)
        self.spike_prob = spike_prob
        self.spike_amplitude = spike_amplitude

    def generate(self, t: int) -> float:
        base = np.sin(2 * np.pi * t / Config.total_samples())

        if np.random.rand() < self.spike_prob:
            spike = self.spike_amplitude
        else:
            spike = 0.0

        return base + spike

    def generate_numpy(self, t_values: np.ndarray) -> np.ndarray:
        return np.array([self.generate(int(t)) for t in t_values])

    def generate_torch(self, t_tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            self.generate_numpy(t_tensor.cpu().numpy()),
            device=t_tensor.device,
            dtype=torch.float,
        )


class NonlinearCompositeGenerator(XFeatureGenerator):
    """
    X11: Standalone nonlinear composite signal.

    Formula
    -------
    X11(t) =
        a1 * sin(2pi t / P1)
        + a2 * cos(2pi t / P2)
        + q * (t_norm^2)
        + k * tanh(4 * t_norm)
        + eps_t

    where:
        t_norm in [-1, 1]
        eps_t is Gaussian noise
    """

    def __init__(
        self,
        name: str = "X11",
        amp_fast: float = 1.3,
        amp_slow: float = 0.9,
        quad_coef: float = 0.8,
        tanh_coef: float = 1.2,
        noise_std: float = 0.2,
    ):
        super().__init__(name)
        self.amp_fast = amp_fast
        self.amp_slow = amp_slow
        self.quad_coef = quad_coef
        self.tanh_coef = tanh_coef
        self.noise_std = noise_std
        self.fast_period = Config.hours_per_day
        self.slow_period = Config.hours_per_week()
        self.total = max(1, Config.total_samples() - 1)

    def _core(self, t_values: np.ndarray) -> np.ndarray:
        t = t_values.astype(float)
        t_norm = 2.0 * (t / self.total) - 1.0

        fast = self.amp_fast * np.sin(2 * np.pi * t / self.fast_period)
        slow = self.amp_slow * np.cos(2 * np.pi * t / self.slow_period)
        quad = self.quad_coef * np.square(t_norm)
        saturating = self.tanh_coef * np.tanh(4.0 * t_norm)

        return fast + slow + quad + saturating

    def generate(self, t: int) -> float:
        base = self._core(np.array([t], dtype=float))[0]
        noise = np.random.normal(0.0, self.noise_std)
        return float(base + noise)

    def generate_numpy(self, t_values: np.ndarray) -> np.ndarray:
        base = self._core(t_values)
        noise = np.random.normal(0.0, self.noise_std, size=t_values.shape[0])
        return base + noise

    def generate_torch(self, t_tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            self.generate_numpy(t_tensor.detach().cpu().numpy()),
            device=t_tensor.device,
            dtype=torch.float,
        )
    