import torch

from core.config import Config
from core.setup import import_itransformer_model

from forcast_model.base_forcast_model import BaseForecastModel

Model = import_itransformer_model()
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
    ):
        super().__init__()

        Model = import_itransformer_model()
        self.config = Config()

        self.seq_len = self.config.seq_len
        self.pred_len = self.config.pred_len

        config = type("Config", (), {})()
        config.task_name = "long_term_forecast"
        config.seq_len = self.seq_len
        config.pred_len = self.pred_len
        config.enc_in = input_dim
        config.dec_in = input_dim
        config.c_out = 1
        config.d_model = d_model
        config.n_heads = n_heads
        config.e_layers = e_layers
        config.d_layers = d_layers
        config.d_ff = d_ff
        config.dropout = dropout
        config.activation = "gelu"
        config.factor = 5
        config.output_attention = False
        config.embed = "timeF"
        config.freq = "h"
        config.use_norm = False
        self.learning_rate = self.config.forcaster_learning_rate
        self.epochs = 1
        config.class_strategy = "projection"
        self.batch_size = self.config.batch_size

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Model(config).to(self.device)

    @classmethod
    def search_space(cls):
       return {
            "d_model": [128, 256],
            "n_heads": [4, 8],
            "e_layers": [2, 3],
            "dropout": [0.1, 0.2],
        }
        # return {
        #     "d_model": [128],
        #     "n_heads": [4],
        #     "e_layers": [2],
        #     "dropout": [0.1],
        # }
    def forward(self, X)-> torch.Tensor:
        B = X.shape[0]
        x_dec = torch.zeros(B, self.pred_len, X.shape[2], device=X.device)
        out = self.model(X, None, x_dec, None)
        return out[..., 0:1]

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
            
    def parameters(self)-> torch.Tensor:
        return self.model.parameters()
    def named_parameters(self):
        return self.model.named_parameters()