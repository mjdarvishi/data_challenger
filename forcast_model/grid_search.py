import itertools
import pandas as pd
import torch
from typing import Type, Tuple, Dict, Any, Union, List

from core.config import Config
from forcast_model.base_forcast_model import BaseForecastModel
from training.forcast_trainer import ForecastTrainer


class GridSearchEngine:
    def __init__(self, model_class: Type[BaseForecastModel]):
        self.model_class = model_class
        self.config = Config()

    def search(self, X_train, Y_train, X_test, Y_test)-> Tuple[BaseForecastModel, float, dict, pd.DataFrame]:

        grid = self.model_class.search_space()

        keys = list(grid.keys())
        values = list(grid.values())

        param_grid = list(itertools.product(*values))

        best_model = None
        best_score = float("inf")
        best_params = None
        results: List[Dict[str, Any]] = []

        for combo in param_grid:
            params = dict(zip(keys, combo))

            # =========================
            # 1. build model
            # =========================
            model = self.model_class(**params)

            # =========================
            # 2. wrap trainer
            # =========================
            trainer = ForecastTrainer(model)

            # =========================
            # 3. train
            # =========================
            trainer.fit(
                X_train,
                Y_train,
                epochs=self.config.grade_search_epochs
            )

            # =========================
            # 4. evaluate
            # =========================
            _, score = trainer.evaluate_pred_mse(X_test, Y_test)

            results.append({**params, "mse": score})
            print(f"GRID SEARCH Tested params: {params}, MSE: {score:.4f}")

            if score < best_score:
                best_score = score
                best_model = model
                best_params = params

        if best_model is None:
            raise RuntimeError("Grid search failed")
        print(f"Best params: {best_params}, Best MSE: {best_score:.4f}")
        return best_model, best_score, best_params, pd.DataFrame(results)