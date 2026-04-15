from dataclasses import dataclass
from typing import Dict, List
from data_generator import (
    YearlyWeeklySineNoiseGenerator,
    TemperatureSeasonalGenerator,
    YearlySineGenerator,
    ConstantGenerator,
    ConstantWithNoiseGenerator,
    TemperatureStructuralGenerator,
)
from data_generator.x_feature_generators import XFeatureGenerator
from core.config import Config, XFeature


class XFeatureRegistery:
    def __init__(self):
        self.generator_registry: Dict[XFeature, XFeatureGenerator] = {}
        self.selected_generators: List[XFeatureGenerator] = []
        self._register_default_generators()
        self.config = Config()

    def select_generators(self, generator_names: List[XFeature]):
        missing = [
            name for name in generator_names if name not in self.generator_registry
        ]
        if missing:
            raise ValueError(f"Unknown generators: {missing}")
        self.selected_generators = [
            self.generator_registry[name] for name in generator_names
        ]

    def _register_default_generators(self):
        self.generator_registry[XFeature.X1] = YearlySineGenerator(
            name=XFeature.X1.value,
        )
        self.generator_registry[XFeature.X2] = ConstantGenerator(
            name=XFeature.X2.value,
        )
        self.generator_registry[XFeature.X3] = YearlyWeeklySineNoiseGenerator(
            name=XFeature.X3.value,
        )
        self.generator_registry[XFeature.X4] = ConstantWithNoiseGenerator(
            name=XFeature.X4.value,
        )
        self.generator_registry[XFeature.X5] = TemperatureSeasonalGenerator(
            name=XFeature.X5.value,
        )
        self.generator_registry[XFeature.X6] = TemperatureStructuralGenerator(
            name=XFeature.X6.value,
        )

    def get_features(self, t: int) -> tuple[float, float]:
        if not self.selected_generators:
            raise ValueError("No generators selected. Call select_generators() first.")

        features = []
        for gen in self.selected_generators:
            features.append(gen.generate(t))
        return tuple(features)
