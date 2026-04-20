import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from core.config import Config
from forcast_model.base_forcast_model import BaseForecastModel


def _import_dlinear_model():
    project_root = Path(__file__).resolve().parents[1]
    repo_dir = project_root / "external_models" / "LTSF-Linear"
    # Isolate model imports to avoid collisions with other external repos.
    sys.path = [p for p in sys.path if "external_models" not in str(p)]
    sys.path.insert(0, str(repo_dir))

    for prefix in ("layers", "models", "utils"):
        for name in list(sys.modules):
            if name == prefix or name.startswith(f"{prefix}."):
                mod = sys.modules.get(name)
                mod_file = getattr(mod, "__file__", "") if mod is not None else ""
                if mod_file:
                    try:
                        mod_path = Path(mod_file).resolve()
                    except OSError:
                        continue
                    if repo_dir not in mod_path.parents:
                        sys.modules.pop(name, None)

    importlib.invalidate_caches()
    module = importlib.import_module("models.DLinear")
    return module.Model


class DLinearForcaster(BaseForecastModel):
    def __init__(
        self,
        input_dim: int = None,
        individual: bool = False,
    ):
        super().__init__()

        self.config = Config()
        self.seq_len = self.config.seq_len
        self.pred_len = self.config.pred_len
        self.input_dim = input_dim or self.config.input_dim
        self.individual = individual

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_cls = _import_dlinear_model()
        model_cfg = SimpleNamespace(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            enc_in=self.input_dim,
            individual=self.individual,
        )
        self.core_model = model_cls(model_cfg)
        self.output_projection = nn.Linear(self.input_dim, 1)

        self.model = nn.ModuleDict(
            {
                "core": self.core_model,
                "output_projection": self.output_projection,
            }
        ).to(self.device)

    @classmethod
    def search_space(cls):
        return {
            "individual": [False, True],
        }

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[-1] != self.input_dim:
            raise RuntimeError(f"Expected {self.input_dim} features, got {X.shape[-1]}")

        out = self.core_model(X)             # [B, P, C]
        return self.output_projection(out)   # [B, P, 1]

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.model.parameters():
            p.requires_grad = True

    @property
    def parameters(self):
        return self.model.parameters()

    @property
    def named_parameters(self):
        return self.model.named_parameters()
