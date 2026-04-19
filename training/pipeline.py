# ============================================================
# Adversarial Data Generation + Forecasting Pipeline
# ============================================================

# We define a synthetic time-series learning system composed of:
#
# ------------------------------------------------------------
# 1. Feature Generation Process
# ------------------------------------------------------------
# At each time step t, we generate a feature vector:
#
#     X(t) = [X1(t), X2(t), ..., Xn(t)]
#
# where each feature Xi(t) is produced by a stochastic or
# deterministic generator:
#
#     Xi(t) ~ Gi(t; φi)
#
# φ = parameters of the feature generators
# (e.g., noise_std, amplitudes, periodicity, etc.)
#
# ------------------------------------------------------------
# 2. Ground Truth Target Function
# ------------------------------------------------------------
# The target signal is defined as a linear combination of features:
#
#     Y(t) = b0 + Σ (bi * Xi(t))
#
# or in vector form:
#
#     Y(t) = b0 + b^T X(t)
#
# where:
#     b0  = bias term
#     b   = true coefficients (fixed, not learned by model)
#
# ------------------------------------------------------------
# 3. Forecast Model
# ------------------------------------------------------------
# A neural network (or any function approximator):
#
#     Ŷ(t) = fθ(X(t))
#
# where:
#     θ = learnable parameters of the forecast model
#
# ------------------------------------------------------------
# 4. Loss Function (MSE)
# ------------------------------------------------------------
# The prediction error is:
#
#     L(θ, φ) = E_t [ ( fθ(Xφ(t)) - Yφ(t) )^2 ]
#
# where:
#     Xφ(t) = generated features
#     Yφ(t) = b0 + b^T Xφ(t)
#
# ------------------------------------------------------------
# 5. Min-Max Adversarial Objective
# ------------------------------------------------------------
# This creates a game between two components:
#
# Forecast Model (θ):
#     θ* = argmin_θ L(θ, φ)
#
# Generator (φ):
#     φ* = argmax_φ L(θ, φ)
#
# Final objective:
#
#     min_θ max_φ E_t [
#         ( fθ(Xφ(t)) - (b0 + b^T Xφ(t)) )^2
#     ]
#
# ------------------------------------------------------------
# 6. Interpretation
# ------------------------------------------------------------
# - The generator produces increasingly complex / noisy / structured
#   feature distributions X(t)
# - The forecast model learns to approximate the mapping X → Y
# - Training forms a competitive adversarial system:
#       generator makes task harder
#       model tries to stay accurate
# ============================================================


from unicodedata import name

from data_generator.x_feature_registery import XFeatureRegistery
from data_generator.generator_model import GeneratorModel
from dataset.dataset_builder import DatasetBuilder
from dataset.equence_builder import SequenceBuilder
from dataset.normalizer import DataNormalizer
from dataset.splitter import TimeSeriesSplitter
from training.generator_trainer import GeneratorTrainer
from training.forcast_trainer import ForecastTrainer
from core.config import Config
import torch
from forcast_model.grid_search import GridSearchEngine
from training.pipline_tracker import PipelineTracker
from time import perf_counter


class BasePipeline:
    def __init__(
        self,
        name: str,
        x_registery: XFeatureRegistery,
        gen_model: GeneratorModel,
        grid_search_engine: GridSearchEngine,
    ):
        self.config = Config()
        self.tracker = PipelineTracker()
        self.normalizer = DataNormalizer()
        self.splitter = TimeSeriesSplitter()
        self.dataset_builder = DatasetBuilder()
        self.gen_trainer = GeneratorTrainer(gen_model)
        self.sequence_builder = SequenceBuilder(
            seq_len=self.config.seq_len,
            pred_len=self.config.pred_len,
        )
        self.grid_search_engine = grid_search_engine
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = name
        self.x_registery = x_registery
        self.gen_model = gen_model
        self.forcast_trainer = None

    def select_best_model(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_test: torch.Tensor,
        Y_test: torch.Tensor,
    ):
        best_model, best_score, best_params, grid_df = self.grid_search_engine.search(
            X_train,
            Y_train,
            X_test,
            Y_test,
        )
        self.forcast_trainer = ForecastTrainer(best_model)
        self.tracker.log_grid_search(
            grid_df=grid_df,
            best_params=best_params,
            best_score=best_score,
        )
        return best_model, best_score

    def _build_normalize_splitet(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        X_raw, Y_raw = self.dataset_builder.build(
                    self.x_registery,
                    self.gen_model,
                    n_samples=self.config.total_samples(),
                )
        # =========================
        # 2. NORMALIZE
        # =========================
        self.normalizer.fit(X_raw, Y_raw)
        self.tracker.meta = {
            "Y_mean": float(self.normalizer.Y_mean.item()),
            "Y_std": float(self.normalizer.Y_std.item()),
        }
        X, Y = self.normalizer.transform(X_raw, Y_raw)

        # =========================
        # 3. SEQUENCE BUILD
        # =========================
        X_seq, Y_seq = self.sequence_builder.build(X, Y)

        # =========================
        # 4. SPLIT
        # =========================
        X_train, Y_train, X_test, Y_test = self.splitter.split(X_seq, Y_seq)

        X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)
        X_test, Y_test = X_test.to(self.device), Y_test.to(self.device)

        return X_train, Y_train, X_test, Y_test, X_raw, Y_raw

    def run_step(self, epoch_num: int):
        step_start = perf_counter()

        # 2. normalize, sequence build, split
        X_train, Y_train, X_test, Y_test, X_raw, Y_raw = self._build_normalize_splitet()
        # =========================
        # 1. TRAIN MODEL
        # =========================
        # Detach Y to prevent gradient flow into generator during model training.
        # Ensures only the forecast model (θ) is updated in this step.
        forecast_start = perf_counter()
        model_losses = self.forcast_trainer.fit(X_train,  Y_train.detach())
        self.forcast_trainer.model.eval_mode()

         # =========================
        # 9. EVALUATION
        # =========================
        pred, mse = self.forcast_trainer.evaluate_pred_mse(X_test, Y_test)
        forecast_time = perf_counter() - forecast_start
        # =========================
        # 6. FREEZE MODEL
        # =========================
        self.forcast_trainer.model.freeze()

        # =====================================================
        # 7. TRAIN GENERATOR (ADVERSARIAL LOOP)
        # =====================================================
        generator_loss = {}
        generator_start = perf_counter()
        for step in range(self.config.generator_epoch):
            # We regenerate the dataset inside the loop because the generator changes the data distribution at every update,
            # so the model must always train against the current distribution rather than a fixed snapshot.
            X_train_adv, Y_train_adv, _, _ ,_,_= self._build_normalize_splitet()
            ges_loss=self.gen_trainer.fit_with_per_sample_mse(X_train_adv, Y_train_adv, self.forcast_trainer)
            generator_loss[step] = ges_loss
            # self._debug(step)
        generator_time = perf_counter() - generator_start
        # =========================
        # 8. UNFREEZE MODEL
        # =========================
        self.forcast_trainer.model.unfreeze()

        # =========================
        # 10. TRACK
        # =========================
        self.tracker.log_step(
            step=epoch_num,
            execution_time=perf_counter() - step_start,
            forecast_time=forecast_time,
            generator_time=generator_time,
            model_losses=model_losses,
            generator_loss=generator_loss,
            gen_model=self.gen_model,
            X_raw=X_raw.detach().cpu(),
            Y_raw=Y_raw.detach().cpu(),
            predictions = pred.detach().cpu().clone() ,
            targets = Y_test.detach().cpu().clone()
        )
        print(
            f"Epoch {epoch_num} | total: {perf_counter() - step_start:.2f}s | "
            f"forecast: {forecast_time:.2f}s | generator: {generator_time:.2f}s | "
            f"Model MSE: {mse:.4f} "
            f"generator_loss: {list(generator_loss.values())[0]:.4f}"
        )

    def run(self):

        # 2. normalize, sequence build, split
        X_train, Y_train, X_test, Y_test, _ , _ = self._build_normalize_splitet()
        # 5. find best model
        self.select_best_model(X_train, Y_train, X_test, Y_test)

        # 6. adversarial training loop
        for epoch in range(self.config.training_epochs):
            self.run_step(epoch)
    
    
    def _debug(self,step:int):
        if step<2:
            for name, param in self.gen_model.named_parameters():
                grad_norm = param.grad.norm().item() if param.grad is not None else 0.0
                print(f"  {name}: grad_norm={grad_norm:.6f}")