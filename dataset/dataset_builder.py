import torch
from data_generator.x_feature_registery import XFeatureRegistery
from data_generator.generator_model import GeneratorModel
from core.config import Config
import numpy as np


class DatasetBuilder:

    def build(
        self,
        x_registry: XFeatureRegistery,
        gen_model: GeneratorModel,
        n_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        t = torch.arange(n_samples, dtype=torch.float32)
        t_np = t.cpu().numpy()

        # ================================
        # Build X features
        # ================================
        feature_context: dict[str, np.ndarray] = {}

        for gen in x_registry.selected_generators:
            feature_context[gen.name] = gen.generate_numpy(t_np)

        for gen in x_registry.selected_generators:
            feature_context[gen.name] = gen.generate_numpy_with_context(
                t_np, feature_context
            )

        x_features = [
            torch.tensor(feature_context[gen.name], dtype=torch.float32)
            for gen in x_registry.selected_generators
        ]

        if not x_features:
            raise ValueError("No generators selected.")

        X = torch.stack(x_features, dim=1)  # [T, F]

        if X.shape[1] != gen_model.num_features:
            raise ValueError("Feature mismatch")

        # ================================
        # Generator supports batched sequence input [B, T, F]
        # ================================
        X_seq = X.unsqueeze(0)  # [1, T, F]

        # ================================
        # Generate Y using the hourly ground-truth generator
        # ================================
        Y_seq = gen_model(X_seq)  # [1, T]

        # ================================
        # Forecaster input format
        # Add time index (important!)
        # ================================
        hours_per_week = Config().hours_per_week()
        hour_idx = (t % hours_per_week).unsqueeze(1)

        X_out = torch.cat([hour_idx, X], dim=1)  # [T, F+1]
        Y_out = Y_seq.squeeze(0)  # [T]

        return X_out, Y_out
