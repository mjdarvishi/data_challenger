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
    def log_step(
        self,
        step: int,
        execution_time: float,
        forecast_time: float,
        generator_time: float,
        model_losses: dict[int, float],
        generator_loss: dict[int, float],
        pred_mse: float,
        train_eval_mse: float | None,
        val_eval_mse: float | None,
        test_eval_mse: float | None,
        gen_model: GeneratorModel,
        X_raw: torch.Tensor,
        Y_raw: torch.Tensor,
        predictions: torch.Tensor = None,
        targets: torch.Tensor = None,
        y_mean: float | None = None,
        y_std: float | None = None,
    ):
        # Extract parameters
        b0 = gen_model.b0.detach().cpu().clone()
        b = gen_model.b.detach().cpu().clone()
        n_features = b.shape[0]
        samples_per_step = len(X_raw)

        data_points = []

        for i in range(len(X_raw)):
            global_time = step * samples_per_step + i
            hour = i % 168

            # X_raw layout is [hour_idx, x1..xN].
            # Generator parameters are taken directly from gen_model for logging.
            x_values = X_raw[i, 1 : 1 + n_features].detach().cpu().tolist()
            y = float(Y_raw[i].item())

            data_points.append(
                DataPoint(
                    global_time=global_time,
                    hour_of_week=hour,
                    x_values=x_values,
                    b0_used=float(b0[hour].item()),
                    b_values=b[:, hour].detach().cpu().tolist(),
                    y=y,
                )
            )

        params = {
            "b0": b0.numpy(),
            "b": b.numpy(),
        }
        # Backward-compatible keys expected by existing dashboard views.
        if n_features >= 1:
            params["b1"] = b[0].numpy()
        if n_features >= 2:
            params["b2"] = b[1].numpy()

        record = StepRecord(
            step=step,
            execution_time=execution_time,
            forecast_time=forecast_time,
            generator_time=generator_time,
            model_losses=model_losses,
            generator_loss=generator_loss,
            pred_mse=float(pred_mse),
            train_eval_mse=float(train_eval_mse) if train_eval_mse is not None else None,
            val_eval_mse=float(val_eval_mse) if val_eval_mse is not None else None,
            test_eval_mse=float(test_eval_mse) if test_eval_mse is not None else None,
            params=params,
            data=data_points,
            predictions=predictions,
            targets=targets,
            y_mean=float(y_mean) if y_mean is not None else None,
            y_std=float(y_std) if y_std is not None else None,
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
                "pred_mse": float(record.pred_mse),
                "train_eval_mse": float(record.train_eval_mse) if record.train_eval_mse is not None else None,
                "val_eval_mse": float(record.val_eval_mse) if record.val_eval_mse is not None else None,
                "test_eval_mse": float(record.test_eval_mse) if record.test_eval_mse is not None else None,
                "params": convert(record.params),
                "predictions": convert(record.predictions),
                "targets": convert(record.targets),
                "Y_mean": float(record.y_mean) if record.y_mean is not None else None,
                "Y_std": float(record.y_std) if record.y_std is not None else None,
                "data": [
                    {
                        "global_time": d.global_time,
                        "hour": d.hour_of_week,
                        "x_values": d.x_values,
                        "y": d.y,
                        "b0": d.b0_used,
                        "b_values": d.b_values,
                        # Keep these aliases so old dashboard code keeps working.
                        "x1": d.x_values[0] if len(d.x_values) > 0 else None,
                        "x2": d.x_values[1] if len(d.x_values) > 1 else None,
                        "b1": d.b_values[0] if len(d.b_values) > 0 else None,
                        "b2": d.b_values[1] if len(d.b_values) > 1 else None,
                    }
                    for d in record.data
                ]
            })

        payload = {
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