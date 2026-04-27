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

            Y_pred = forcast_trainer.model.forward(X_train)

            loss = forcast_trainer.criterion(Y_pred, Y_train)
            self.optimizer.zero_grad()
            (-loss).backward()   # loss is already negated inside adversarial_loss
            self.optimizer.step()
            # self.gen_model.clamp_parameters()

            losses[step] = float(forcast_trainer.criterion(Y_pred, Y_train).item())

        return losses
