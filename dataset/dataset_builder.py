import torch
from data_generator.x_feature_registery import XFeatureRegistery
from data_generator.generator_model import GeneratorModel
from core.config import Config


class DatasetBuilder:
    
    def build(
        self, x_registry: XFeatureRegistery, gen_model: GeneratorModel, n_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        X_list = []
        Y_list = []

        t = torch.arange(n_samples, dtype=torch.float32)

        x_features = [gen.generate_torch(t) for gen in x_registry.selected_generators]

        x1, x2 = x_features

        hours_per_week = Config().hours_per_week()

        for i in range(n_samples):
            hour_idx = i % hours_per_week

            y = gen_model(hour_idx, x1[i], x2[i])

            # Keep the same 6-channel layout used in the original adversarial pipeline.
            X_list.append(
                torch.stack(
                    [
                        torch.tensor(float(hour_idx), dtype=torch.float32),
                        x1[i],
                        x2[i],
                        gen_model.b0[hour_idx],
                        gen_model.b1[hour_idx],
                        gen_model.b2[hour_idx],
                    ]
                )
            )
            Y_list.append(y)

        X = torch.stack(X_list)
        Y = torch.stack(Y_list)

        return X, Y
