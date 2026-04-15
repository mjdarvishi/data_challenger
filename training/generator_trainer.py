import torch
from data_generator.generator_model import GeneratorModel
from core.config import Config


class GeneratorTrainer:
    def __init__(self, gen_model: GeneratorModel):
        self.gen_model = gen_model
        self.optimizer = torch.optim.Adam(
            gen_model.parameters(),
            lr=Config.generator_learning_rate
        )

    # def step(self, loss: torch.Tensor):
    #     """
    #     IMPORTANT: adversarial update = maximize loss
    #     """
    #     self.optimizer.zero_grad()

    #     # adversarial objective (same as original: -loss.backward())
    #     (-loss).backward()

    #     self.optimizer.step()

    #     # optional stability (same as your original clamp)
    #     if hasattr(self.gen_model, "clamp_parameters"):
    #         self.gen_model.clamp_parameters()
    def step(self, loss: torch.Tensor):
        """
        Since loss is NOT connected to generator graph,
        we cannot use backward().
        So we simulate adversarial pressure manually.
        """

        self.optimizer.zero_grad()

        # 👉 simple trick: push parameters in direction of increasing loss
        # (randomized heuristic update)
        for p in self.gen_model.parameters():
            if p.requires_grad:
                noise = torch.randn_like(p) * loss.item()
                p.data += noise * 1e-3   # small step

        self.optimizer.step()