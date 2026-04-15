import torch
from torch import nn
from core.config import Config
from forcast_model.base_forcast_model import BaseForecastModel
from torch.utils.data import DataLoader


class ForcasterTrainer:
    def __init__(self, model: BaseForecastModel):
        self.model = model
        self.config = Config()
        self.criterion = nn.MSELoss()
        

    def train(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        self.model.train_mode()

        last_loss = None

        dataset = list(zip(X, Y))
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        for _ in range(self.config.forcast_trainer_epoch):
            for x_batch, y_batch in loader:
                last_loss = self.model.train_step(x_batch, y_batch, self.criterion)

        return last_loss


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model.forward(X)