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
    
    def fit_with_per_sample_mse(self, X: torch.Tensor, Y: torch.Tensor, forcast_trainer: ForecastTrainer) -> torch.Tensor:
        self.optimizer.zero_grad()
        
        forcast_trainer.model.eval_mode()
        preds = forcast_trainer.model.forward(X)
        
        if Y.dim() == 2:
            Y = Y.unsqueeze(-1)
        
        # Per-sample MSE
        per_sample_mse = (preds - Y).pow(2).mean(dim=(1, 2))  # [B]
        
        # Option A: maximize mean (your current approach, weak)
        # loss = -per_sample_mse.mean()
        
        # Option B: maximize top-k hardest samples (much stronger)
        k = max(1, int(0.3 * len(per_sample_mse)))  # hardest 30%
        hard_mse, _ = per_sample_mse.topk(k)
        loss = -hard_mse.mean()
        
        loss.backward()
        self.optimizer.step()
        self.gen_model.clamp_parameters()
        self.optimizer.zero_grad()
        return -loss.item()