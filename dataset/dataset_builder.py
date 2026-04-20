import torch
from data_generator.x_feature_registery import XFeatureRegistery
from data_generator.generator_model import GeneratorModel
from core.config import Config
import numpy as np


class DatasetBuilder:
    
    def build(
        self, x_registry: XFeatureRegistery, gen_model: GeneratorModel, n_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        X_list = []
        Y_list = []

        t = torch.arange(n_samples, dtype=torch.float32)
        t_np = t.detach().cpu().numpy()

        # Pass 1: build base features independently.
        feature_context: dict[str, np.ndarray] = {}
        for gen in x_registry.selected_generators:
            feature_context[gen.name] = gen.generate_numpy(t_np)

        # Pass 2: allow generators to refine output using other X features.
        for gen in x_registry.selected_generators:
            feature_context[gen.name] = gen.generate_numpy_with_context(t_np, feature_context)

        x_features = [
            torch.tensor(feature_context[gen.name], dtype=torch.float32)
            for gen in x_registry.selected_generators
        ]
        if not x_features:
            raise ValueError("No generators selected. Call select_generators() first.")

        x_matrix = torch.stack(x_features, dim=1)
        n_features = x_matrix.shape[1]
        if n_features != gen_model.num_features:
            raise ValueError(
                f"GeneratorModel expects {gen_model.num_features} features, got {n_features}"
            )

        hours_per_week = Config().hours_per_week()

        for i in range(n_samples):
            hour_idx = i % hours_per_week
            x_row = x_matrix[i]

            y = gen_model(hour_idx, x_row)

            hour_tensor = torch.tensor([float(hour_idx)], dtype=torch.float32)

            # Forecaster sees only observable inputs; generator params stay hidden.
            # Layout: [hour_idx, x1..xN]
            X_list.append(torch.cat([hour_tensor, x_row]))
            Y_list.append(y)

        X = torch.stack(X_list)
        Y = torch.stack(Y_list)

        return X, Y
