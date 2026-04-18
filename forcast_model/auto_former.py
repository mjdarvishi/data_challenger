import torch
from types import SimpleNamespace

from core.config import Config
from forcast_model.base_forcast_model import BaseForecastModel
from core.setup import import_autoformer_model


class AutoformerForcaster(BaseForecastModel):
    def __init__(
        self,
        d_model=128,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        d_ff=512,
        dropout=0.1,
    ):
        super().__init__()

        self.config = Config()
        self.input_dim = self.config.input_dim

        self.seq_len = self.config.seq_len
        self.pred_len = self.config.pred_len
        self.label_len = self.seq_len // 2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # =========================
        # REAL AUTOFORMER CONFIG (FIXED)
        # =========================
        cfg = SimpleNamespace()

        cfg.task_name = "long_term_forecast"

        cfg.seq_len = self.seq_len
        cfg.label_len = self.label_len
        cfg.pred_len = self.pred_len

        cfg.enc_in = self.input_dim
        cfg.dec_in = self.input_dim
        cfg.c_out = 1

        cfg.d_model = d_model
        cfg.n_heads = n_heads
        cfg.e_layers = e_layers
        cfg.d_layers = d_layers
        cfg.d_ff = d_ff
        cfg.dropout = dropout

        cfg.output_attention = False
        cfg.embed = "timeF"
        cfg.freq = "h"

        cfg.factor = 3
        cfg.moving_avg = 25
        cfg.activation = "gelu"

        model_cls = import_autoformer_model()
        self.model = model_cls(cfg).to(self.device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[-1] != self.input_dim:
            raise RuntimeError(
                f"Expected {self.input_dim} features, got {X.shape[-1]}"
            )

        B = X.shape[0]

        dec_inp = torch.zeros(
            B,
            self.label_len + self.pred_len,
            self.input_dim,
            device=X.device,
        )

        x_mark_enc = torch.zeros(B, self.seq_len, 4, device=X.device)
        x_mark_dec = torch.zeros(B, self.label_len + self.pred_len, 4, device=X.device)

        out = self.model(X, x_mark_enc, dec_inp, x_mark_dec)

        return out[..., :1]

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

    @classmethod
    def search_space(cls):
        return {
            "d_model": [128, 256],
            "n_heads": [4, 8],
            "e_layers": [2, 3],
            "dropout": [0.1, 0.2],
        }