import torch
import numpy as np

from core.config import Config
from core.setup import import_samformer_class
from forcast_model.base_forcast_model import BaseForecastModel

SAMFormer = import_samformer_class()


class SamformerForcaster(BaseForecastModel):
    def __init__(
        self,
        input_dim=2,
        learning_rate=1e-3,
        weight_decay=1e-5,
        rho=0.5,
        use_revin=True,
        random_state=42,
    ):
        super().__init__()

        self.config = Config()

        self.seq_len = self.config.seq_len
        self.pred_len = self.config.pred_len
        self.batch_size = self.config.batch_size

        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.rho = rho
        self.use_revin = use_revin
        self.random_state = random_state

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SAMFormer(
            device=self.device,
            num_epochs=1,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            rho=self.rho,
            use_revin=self.use_revin,
            random_state=self.random_state,
        )

    @classmethod
    def search_space(cls):
        return {
            "learning_rate": [1e-3, 5e-4],
            "rho": [0.3, 0.5],
            "use_revin": [True, False],
        }

    @staticmethod
    def _to_samformer_layout(X: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, C, T)
        return X.transpose(1, 2).contiguous()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        SAMFormer is not a torch nn.Module (fit/predict style),
        so forward = predict
        """
        X_sam = self._to_samformer_layout(X).detach().cpu()

        y_pred = self.model.predict(X_sam)

        if isinstance(y_pred, np.ndarray):
            y_pred = torch.tensor(y_pred, dtype=torch.float32)

        #  Take first channel → scalar forecast
        y_scalar = y_pred[:, 0].unsqueeze(-1).to(X.device)

        #  Expand to (B, pred_len, 1)
        return y_scalar.unsqueeze(1).repeat(1, self.pred_len, 1)

    def train_mode(self):
        pass  # SAMFormer handles internally

    def eval_mode(self):
        pass

    def freeze(self):
        pass

    def unfreeze(self):
        pass

    @property
    def parameters(self):
        return []  # Not a torch model

    @property
    def named_parameters(self):
        return []