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
    ) -> dict[int, dict[str, float]]:
        losses = {}

        for step in range(self.config.generator_epoch):
            split: "PipelineSplitResult" = build_normalize_splitet_func()
            X_train, Y_train = self._sample_batch(split.X_train, split.Y_train)
            X_val, Y_val = self._sample_batch(split.X_val, split.Y_val)
            X_test, Y_test = self._sample_batch(split.X_test, split.Y_test)

            # X_train contains historical Y, so allow generator gradients through
            # that history channel as well as the future Y target.
            train_loss = self._forecast_loss(
                forcast_trainer,
                X_train,
                Y_train,
            )
            val_loss = self._forecast_loss(
                forcast_trainer,
                X_val,
                Y_val,
            )
            test_loss = self._forecast_loss(
                forcast_trainer,
                X_test,
                Y_test,
            )
            forecast_loss = (
                self.config.generator_train_loss_weight * train_loss
                + self.config.generator_val_loss_weight * val_loss
                + self.config.generator_test_loss_weight * test_loss
            )
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

            losses[step] = {
                "weighted": float(forecast_loss.item()),
                "train": float(train_loss.item()),
                "val": float(val_loss.item()),
                "test": float(test_loss.item()),
            }

        return losses

    def _forecast_loss(
        self,
        forcast_trainer: ForecastTrainer,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        Y_pred = forcast_trainer.model.forward(X)
        target = Y
        if target.dim() == Y_pred.dim() - 1:
            target = target.unsqueeze(-1)

        return forcast_trainer.criterion(Y_pred, target)

    def _sample_batch(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = min(int(self.config.batch_size), X.shape[0])
        indices = torch.randperm(X.shape[0], device=X.device)[:batch_size]
        return X.index_select(0, indices), Y.index_select(0, indices)
