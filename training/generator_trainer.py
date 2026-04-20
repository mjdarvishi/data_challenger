import torch
from data_generator.generator_model import GeneratorModel
from core.config import Config
from training.forcast_trainer import ForecastTrainer
import torch.nn.functional as F


class GeneratorTrainer:
    def __init__(self, gen_model: GeneratorModel):
        self.gen_model = gen_model
        self.optimizer = torch.optim.Adam(
            gen_model.parameters(), lr=Config.generator_trainer_learning_rate
        )
        self.config = Config()

    def fit(self, forcast_trainer: ForecastTrainer, build_normalize_splitet_func: callable) -> dict[int, float]:
        generator_loss: dict[int, float] = {}
        for step in range(self.config.generator_epoch):
            X_train, Y_train, _, _ ,_,_= build_normalize_splitet_func()
            Y_pred = forcast_trainer.model.forward(X_train)
            loss: torch.Tensor =  forcast_trainer.criterion(Y_pred, Y_train)
            
            generator_loss[step] = float(loss.item())

            self.optimizer.zero_grad()
            (-loss).backward()
            self.optimizer.step() 
            self.gen_model.clamp_parameters()
            generator_loss[step] = float(loss.item())
            
        return generator_loss
    
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
    
    def fit_with_diversity_penalty(self, X_train: torch.Tensor, Y_train: torch.Tensor, forecast_trainer: ForecastTrainer):
        self.optimizer.zero_grad()
        
        preds = forecast_trainer.model.forward(X_train)
        target = Y_train
        if target.dim() == 2:
            target = target.unsqueeze(-1)
        
        adversarial_loss = -F.mse_loss(preds, target)
        
        # Penalize the generator for making all hours look the same
        # Forces it to find diverse hard patterns, not just one extreme
        b_stack = torch.cat([
            self.gen_model.b0.unsqueeze(0),
            self.gen_model.b,
        ], dim=0)
        
        # Maximize variance across hours (diversity reward)
        diversity_bonus = -b_stack.var(dim=1).mean() * 0.1
        
        loss = adversarial_loss + diversity_bonus
        loss.backward()
        self.optimizer.step()
        self.gen_model.clamp_parameters()
        
        return -adversarial_loss.item()