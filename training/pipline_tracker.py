import pandas as pd
import torch

from core.models import DataPoint, StepRecord
from data_generator.generator_model import GeneratorModel


class PipelineTracker:
    def __init__(self):
        self.history: list[StepRecord] = []
        self.grid_search_history = []

    def log_step(
        self,
        step: int,
        model_loss: float,
        generator_loss: float,
        gen_model: GeneratorModel,
        X_raw: torch.Tensor,
        Y_raw: torch.Tensor,
      predictions: torch.Tensor =None,
    targets: torch.Tensor =None,
    ):
        # Extract parameters
        b0 = gen_model.b0.detach().cpu()
        b1 = gen_model.b1.detach().cpu()
        b2 = gen_model.b2.detach().cpu()

        data_points = []

        for i in range(len(X_raw)):
            hour = i % 168

            x1 = float(X_raw[i, 0].item())
            x2 = float(X_raw[i, 1].item())
            y = float(Y_raw[i].item())

            data_points.append(
                DataPoint(
                    hour_of_week=hour,
                    x1=x1,
                    x2=x2,
                    b0_used=float(b0[hour].item()),
                    b1_used=float(b1[hour].item()),
                    b2_used=float(b2[hour].item()),
                    y=y,
                )
            )

        record = StepRecord(
            step=step,
            model_loss=model_loss,
            generator_loss=generator_loss,
            params={
                "b0": b0.numpy(),
                "b1": b1.numpy(),
                "b2": b2.numpy(),
            },
            data=data_points,
            predictions=predictions,
            targets=targets,
        )

        self.history.append(record)

    def log_grid_search(
        self,
        grid_df: pd.DataFrame,
        best_params: dict,
        best_score: float,
    ):
        """
        Store full grid search results + best configuration.
        """

        self.grid_search_history.append(
            {
                "results": grid_df,
                "best_params": best_params,
                "best_score": float(best_score),
            }
        )
