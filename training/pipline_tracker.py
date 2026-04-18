import json
import os
from pathlib import Path
import pandas as pd
from core.config import Config
import torch
from core.models import DataPoint, StepRecord
from data_generator.generator_model import GeneratorModel


class PipelineTracker:
    def __init__(self):
        self.history: list[StepRecord] = []
        self.grid_search_history = []
        self.output_dir = self.get_output_dir()

        self.meta = {}
    def log_step(
        self,
        step: int,
        execution_time: float,
        forecast_time: float,
        generator_time: float,
        model_losses: dict[int, float],
        generator_loss: dict[int, float],
        gen_model: GeneratorModel,
        X_raw: torch.Tensor,
        Y_raw: torch.Tensor,
        predictions: torch.Tensor = None,
        targets: torch.Tensor = None,
    ):
        # Extract parameters
        b0 = gen_model.b0.detach().cpu().clone()
        b1 = gen_model.b1.detach().cpu().clone()
        b2 = gen_model.b2.detach().cpu().clone()
        samples_per_step = len(X_raw)

        data_points = []

        for i in range(len(X_raw)):
            global_time = step * samples_per_step + i
            hour = i % 168
            
            # X_raw layout is [hour_idx, x1, x2, b0, b1, b2].
            x1 = float(X_raw[i, 1].item())
            x2 = float(X_raw[i, 2].item())
            y = float(Y_raw[i].item())

            data_points.append(
                DataPoint(
                    global_time=global_time,
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
            execution_time=execution_time,
            forecast_time=forecast_time,
            generator_time=generator_time,
            model_losses=model_losses,
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
        records = []
        if isinstance(grid_df, pd.DataFrame) and not grid_df.empty:
            # Keep rows JSON-ready right away so the dashboard does not depend on late conversion.
            for row in grid_df.to_dict(orient="records"):
                normalized = {}
                for key, value in row.items():
                    if isinstance(value, (int, float, str, bool)) or value is None:
                        normalized[key] = value
                    else:
                        try:
                            normalized[key] = float(value)
                        except (TypeError, ValueError):
                            normalized[key] = str(value)
                records.append(normalized)

        self.grid_search_history.append(
            {
                "results": records,
                "best_params": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in best_params.items()},
                "best_score": float(best_score),
                "num_tested": len(records),
            }
        )

    def export(self, name: str = "dashboard_data.json"):
        path = f"{self.output_dir}/{name}.json"
        def convert(obj):
            import numpy as np
            import torch

            if isinstance(obj, pd.DataFrame):
                return convert(obj.to_dict(orient="records"))

            if isinstance(obj, np.ndarray):
                return obj.tolist()

            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()

            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()

            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}

            if isinstance(obj, list):
                return [convert(x) for x in obj]

            return obj

        data = []

        for record in self.history:
            data.append({
                "step": record.step,
                "execution_time": record.execution_time,
                "forecast_time": record.forecast_time,
                "generator_time": record.generator_time,
                "model_losses": convert(record.model_losses),
                "generator_loss": convert(record.generator_loss),
                "params": convert(record.params),
                "predictions": convert(record.predictions),
                "targets": convert(record.targets),
                "data": [
                    {
                        "global_time": d.global_time,
                        "hour": d.hour_of_week,
                        "x1": d.x1,
                        "x2": d.x2,
                        "y": d.y,
                        "b0": d.b0_used,
                        "b1": d.b1_used,
                        "b2": d.b2_used,
                    }
                    for d in record.data
                ]
            })

        payload = {
            "meta": convert(self.meta),
            "records": data,
            "grid_search_history": convert(self.grid_search_history),
            "config": Config.to_dict(),
        }

        import json
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
            
                
    def get_output_dir(self):
        # 1. detect running environment safely
        if "COLAB_GPU" in os.environ:
            base = Path("/content/data_challenger")
        else:
            base = Path(__file__).resolve().parent.parent

        output_dir = base / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir