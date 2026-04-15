import torch

from core.config import Config
from core.setup import import_itransformer_model

from forcast_model.base_forcast_model import BaseForecastModel

Model = import_itransformer_model()
import torch.nn as nn


class ITransformerWrapper(BaseForecastModel):
    def __init__(
        self,
        input_dim=2,
        d_model=128,
        n_heads=4,
        e_layers=2,
        dropout=0.1,
        d_layers=1,
        d_ff=512,
        learning_rate=1e-4,
        epochs=1,
        batch_size=32,
    ):
        super().__init__()
        Model = import_itransformer_model()
        self.config = Config()
        self.seq_len = self.config.seq_len
        self.pred_len = self.config.pred_len
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.dropout = dropout
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        config = type("Config", (), {})()
        config.task_name = "long_term_forecast"
        config.seq_len = self.seq_len
        config.pred_len = self.pred_len
        config.enc_in = self.input_dim
        config.dec_in = self.input_dim
        config.c_out = 1
        config.d_model = self.d_model
        config.n_heads = self.n_heads
        config.e_layers = self.e_layers
        config.d_layers = self.d_layers
        config.d_ff = self.d_ff
        config.dropout = self.dropout
        config.activation = "gelu"
        config.factor = 5
        config.output_attention = False
        config.embed = "timeF"
        config.freq = "h"
        config.use_norm = False
        config.class_strategy = "projection"

        self.device = self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Model(config).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()

    @classmethod
    def search_space(cls):
        # return {
        #     "d_model": [128, 256],
        #     "n_heads": [4, 8],
        #     "e_layers": [2, 3],
        #     "dropout": [0.1, 0.2],
        # }
        return {
                    "d_model": [128],
                    "n_heads": [4],
                    "e_layers": [2],
                    "dropout": [0.1],
                }
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

    def forward(self, X):
        B = X.shape[0]
        x_dec = torch.zeros(B, self.pred_len, X.shape[2], device=X.device)

        out = self.model(X, None, x_dec, None)
        return out[..., 0:1]

    def train_step(self, X, Y, criterion):
        self.optimizer.zero_grad()

        preds = self.forward(X)
        loss = criterion(preds, Y)

        loss.backward()
        self.optimizer.step()

        return loss

    def evaluate(self, X_test, Y_test) -> float:
        self.eval_mode()

        with torch.no_grad():
            preds = self.forward(X_test).squeeze(-1)
            target = Y_test.squeeze(-1)
            mse = self.criterion(preds, target).item()

        return mse

    def fit(self, X_train, Y_train):
        self.model.train()

        X_train = X_train.detach()
        Y_train = Y_train.detach()

        for _ in range(self.epochs):
            self.optimizer.zero_grad()

            preds = self.forward(X_train)
            loss = self.criterion(preds, Y_train)

            loss.backward()
            self.optimizer.step()

        return loss