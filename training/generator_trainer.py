import torch
from data_generator.generator_model import GeneratorModel
from core.config import Config
from training.forcast_trainer import ForecastTrainer


class GeneratorTrainer:
    def __init__(self, gen_model: GeneratorModel):
        self.gen_model = gen_model
        self.optimizer = torch.optim.Adam(
            gen_model.parameters(), lr=Config.generator_learning_rate
        )

    def fit(self, X: torch.Tensor, Y: torch.Tensor,forcast_trainer: ForecastTrainer) -> float:
        y_pred_adv = forcast_trainer.model.forward(X)
        adv_loss =  forcast_trainer.criterion(y_pred_adv, Y)
        
        self.optimizer.zero_grad()
        (-adv_loss).backward()
        self.optimizer.step() 
        self.gen_model.clamp_parameters()
        return float(adv_loss.item())