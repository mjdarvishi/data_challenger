import json
import os
from pathlib import Path
import pandas as pd
import torch
import numpy as np

from core.config import Config
from core.models import DataPoint, StepRecord
from data_generator.generator_model import GeneratorModel


class PipelineTracker:
    def __init__(self):
        self.history: list[StepRecord] = []
        self.grid_search_history = []
        self.output_dir = self.get_output_dir()

    # =========================================================
    # SAFE TENSOR CONVERTER
    # =========================================================
    def _safe(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().clone()
        return x

    # =========================================================
    # MAIN LOGGER
    # =========================================================
    def log_step(
        self,
        step: int,
        execution_time: float,
        forecast_time: float,
        generator_time: float,
        model_losses: dict,
        generator_loss: dict,
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

        samples_per_step = len(X_raw)
        n_features = X_raw.shape[1] - 1  # exclude hour column

        data_points = []

        # =========================================================
        # PER-SAMPLE LOGGING
        # =========================================================
        for i in range(len(X_raw)):

            global_time = step * samples_per_step + i

            # FIXED: no hardcoded 168
            hour = int(X_raw[i, 0].item())

            x_values = X_raw[i, 1:].detach().cpu().tolist()
            y = float(Y_raw[i].item())

            b0_used = None
            b_values = None
            if hasattr(gen_model, "b0") and hasattr(gen_model, "b"):
                hour_idx = hour % Config.hours_per_week()
                b0_used = float(gen_model.b0[hour_idx].detach().cpu().item())
                b_values = gen_model.b[:, hour_idx].detach().cpu().tolist()

            data_points.append(
                DataPoint(
                    global_time=global_time,
                    hour_of_week=hour,
                    x_values=x_values,
                    b0_used=b0_used,
                    b_values=b_values,
                    y=y,
                )
            )

        # =========================================================
        # NEURAL GENERATOR PARAM SNAPSHOT
        # =========================================================
        params = self._extract_generator_params(gen_model)

        # =========================================================
        # SAFE OUTPUT TENSORS
        # =========================================================
        predictions = self._safe(predictions)
        targets = self._safe(targets)

        # =========================================================
        # RECORD
        # =========================================================
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

    # =========================================================
    # EXTRACT NEURAL GENERATOR PARAMETERS
    # =========================================================
    def _extract_generator_params(self, gen_model: GeneratorModel):
        def safe_state_dict(module):
            return {
                k: self._safe(v).numpy().tolist()
                for k, v in module.state_dict().items()
            }

        params = {}

        if hasattr(gen_model, "b0"):
            params["b0"] = self._safe(gen_model.b0).numpy().tolist()

        if hasattr(gen_model, "b"):
            params["b"] = self._safe(gen_model.b).numpy().tolist()

        if hasattr(gen_model, "residual_scale"):
            params["residual_scale"] = float(
                self._safe(gen_model.residual_scale).numpy().item()
            )

        if hasattr(gen_model, "residual_encoder"):
            params["residual_encoder"] = safe_state_dict(gen_model.residual_encoder)

        if hasattr(gen_model, "temporal_filter"):
            params["temporal_filter"] = safe_state_dict(gen_model.temporal_filter)

        if hasattr(gen_model, "residual_head"):
            params["residual_head"] = safe_state_dict(gen_model.residual_head)

        # Encoder
        if hasattr(gen_model, "encoder"):
            params["encoder"] = safe_state_dict(gen_model.encoder)

        # GRU memory
        if hasattr(gen_model, "rnn"):
            params["rnn"] = safe_state_dict(gen_model.rnn)

        # Experts (MoE)
        if hasattr(gen_model, "experts"):
            params["experts"] = [
                safe_state_dict(expert) for expert in gen_model.experts
            ]

        # Gate network
        if hasattr(gen_model, "gate"):
            params["gate"] = safe_state_dict(gen_model.gate)

        return params

    # =========================================================
    # GRID SEARCH LOGGING
    # =========================================================
    def log_grid_search(
        self,
        grid_df: pd.DataFrame,
        best_params: dict,
        best_score: float,
    ):
        records = []

        if isinstance(grid_df, pd.DataFrame) and not grid_df.empty:
            for row in grid_df.to_dict(orient="records"):
                cleaned = {}
                for k, v in row.items():
                    try:
                        cleaned[k] = float(v)
                    except Exception:
                        cleaned[k] = v
                records.append(cleaned)

        self.grid_search_history.append(
            {
                "results": records,
                "best_params": best_params,
                "best_score": float(best_score),
                "num_tested": len(records),
            }
        )

    # =========================================================
    # EXPORT FULL PIPELINE DATA
    # =========================================================
    def export(self, name: str = "dashboard_data.json"):

        path = f"{self.output_dir}/{name}.json"

        def convert(obj):
            if torch.is_tensor(obj):
                return obj.detach().cpu().tolist()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
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
                "train_eval_mse": float(record.train_eval_mse) if record.train_eval_mse else None,
                "val_eval_mse": float(record.val_eval_mse) if record.val_eval_mse else None,
                "test_eval_mse": float(record.test_eval_mse) if record.test_eval_mse else None,
                "params": convert(record.params),
                "predictions": convert(record.predictions),
                "targets": convert(record.targets),
                "Y_mean": record.y_mean,
                "Y_std": record.y_std,
                "data": [
                    {
                        "global_time": d.global_time,
                        "hour": d.hour_of_week,
                        "x_values": d.x_values,
                        "y": d.y,
                    }
                    for d in record.data
                ]
            })

        payload = {
            "records": data,
            "grid_search_history": convert(self.grid_search_history),
            "config": Config.to_dict(),
        }

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    # =========================================================
    # OUTPUT DIRECTORY
    # =========================================================
    def get_output_dir(self):
        if "COLAB_GPU" in os.environ:
            base = Path("/content/data_challenger")
        else:
            base = Path(__file__).resolve().parent.parent

        output_dir = base / "output/new"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
