import itertools
import pandas as pd
import torch
from typing import Type, Tuple, Dict, Any, Union, List

from core.config import Config
from forcast_model.base_forcast_model import BaseForecastModel


class GridSearchEngine:
    def __init__(
        self,
        model_class: Type[BaseForecastModel],
    ):
        self.model_class = model_class
        self.config = Config()

    def search(
        self,
        X_train: Union[torch.Tensor, Any],
        Y_train: Union[torch.Tensor, Any],
        X_test: Union[torch.Tensor, Any],
        Y_test: Union[torch.Tensor, Any],
    ) -> Tuple[BaseForecastModel, float, Dict[str, Any], pd.DataFrame]:

        grid: Dict[str, List[Any]] = self.model_class.search_space()

        keys = list(grid.keys())
        values = list(grid.values())

        param_grid = list(itertools.product(*values))

        best_model: BaseForecastModel | None = None
        best_score: float = float("inf")
        results: List[Dict[str, Any]] = []

        for combo in param_grid:
            params = dict(zip(keys, combo))

            model = self.model_class(**params)

            model.fit(X_train, Y_train)
            score = model.evaluate(X_test, Y_test)

            results.append({**params, "mse": float(score)})

            if score < best_score:
                best_score = float(score)
                best_model = model
                best_params = params

        if best_model is None:
            raise RuntimeError("Grid search failed: no valid model found.")

        return best_model, best_score, best_params, pd.DataFrame(results)
