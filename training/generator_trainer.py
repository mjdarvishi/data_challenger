import torch
from data_generator.generator_model import GeneratorModel
from core.config import Config
from training.forcast_trainer import ForecastTrainer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training.pipeline import PipelineSplitResult


class GeneratorTrainer:
    def __init__(self, gen_model: GeneratorModel):
        self.gen_model = gen_model
        self.optimizer = torch.optim.Adam(
            gen_model.parameters(), lr=Config.generator_trainer_learning_rate
        )
        self.config = Config()

    def fit(
        self,
        forcast_trainer: ForecastTrainer,
        build_normalize_splitet_func: callable,
    ) -> dict[int, float]:
        losses = {}

        for step in range(self.config.generator_epoch):
            split: "PipelineSplitResult" = build_normalize_splitet_func()
            X_train, Y_train = split.X_train, split.Y_train

            # X_train contains historical Y, so allow generator gradients through
            # that history channel as well as the future Y target.
            Y_pred = forcast_trainer.model.forward(X_train)

            forecast_loss = forcast_trainer.criterion(Y_pred, Y_train)
            realism_loss = self.gen_model.regularization_loss()
            generator_loss = -forecast_loss + self.config.generator_realism_weight * realism_loss

            self.optimizer.zero_grad()
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.gen_model.parameters(),
                self.config.generator_grad_clip,
            )
            self.optimizer.step()
            self.gen_model.clamp_parameters()

            losses[step] = float(forecast_loss.item())

        return losses
