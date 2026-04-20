import torch
from torch import nn
from core.config import Config
from forcast_model.base_forcast_model import BaseForecastModel
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

class ForecastTrainer:
    def __init__(self, model: BaseForecastModel):
        self.model = model
        self.config = Config()
        self.optimizer = torch.optim.Adam(
            self.model.parameters, lr=self.config.forcaster_trainer_learning_rate
        )
        self.criterion = nn.MSELoss()

    def train_step(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:

        self.optimizer.zero_grad()

        preds = self.model.forward(X)
        loss = self.criterion(preds, Y)

        loss.backward()
        self.optimizer.step()

        return loss
    
    def predict_with_squeezed_output(self, X: torch.Tensor) -> torch.Tensor:
        preds = self.model.forward(X)
        return preds.squeeze(-1)

    def fit(self, X_train: torch.Tensor, Y_train: torch.Tensor, epochs: int = None) -> dict[int, float]:
        self.model.train_mode()

        epochs = epochs or self.config.forcast_trainer_epoch
        X_train = X_train.detach()
        Y_train = Y_train.detach()

        losts = {}

        train_loader = DataLoader(
            list(zip(X_train, Y_train)),
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        

        
        for step in range(epochs):
            total_loss = 0
            count = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(X_train.device)
                y_batch = y_batch.to(Y_train.device)
                loss  = self.train_step(x_batch, y_batch)
                total_loss += loss.item()
                count += 1

            losts[step] = total_loss / count

        return losts

    def evaluate_pred_mse(self, X_test: torch.Tensor, Y_test: torch.Tensor) -> tuple[torch.Tensor, float]:
        self.model.eval_mode()

        with torch.no_grad():
            preds = self.model.forward(X_test)

            # IMPORTANT: force same shape
            target = Y_test

            if target.dim() == 2:
                target = target.unsqueeze(-1)

            mse = self.criterion(preds, target).item()

        return preds, mse
    
    
    
# def fit(self, X_train: torch.Tensor, Y_train: torch.Tensor, epochs: int = None) -> dict[int, float]:
#         self.model.train_mode()
    
#         epochs = epochs or self.config.forcast_trainer_epoch
#         X_train = X_train.detach()
#         Y_train = Y_train.detach()

#         losts = {}

#         for step in range(self.config.forcast_trainer_epoch):
#             last_loss = self.train_step(X_train, Y_train)
#             losts[step] = float(last_loss.item())

#         return losts