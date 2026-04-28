import torch
import torch.nn as nn

from core.config import Config


class DependencyAwareFeatureSelector(nn.Module):
    def __init__(
        self,
        num_features: int,
        feature_names: list[str],
        feature_dependencies: dict[str, list[str]] | None = None,
    ):
        super().__init__()
        if num_features <= 0:
            raise ValueError("DependencyAwareFeatureSelector requires at least one feature")
        if len(feature_names) != num_features:
            raise ValueError("feature_names length must match num_features")

        self.config = Config()
        self.num_features = num_features
        self.feature_names = feature_names
        self.feature_dependencies = feature_dependencies or {}
        self.feature_logits = nn.Parameter(
            torch.randn(num_features, dtype=torch.float32) * 0.01
        )
        self._dependency_indices = self._build_dependency_indices()

    def probabilities(self) -> torch.Tensor:
        temperature = max(float(self.config.generator_feature_selection_temperature), 1e-3)
        return torch.sigmoid(self.feature_logits / temperature)

    def gates(self) -> torch.Tensor:
        probs = self.probabilities()
        top_k = self.top_k()

        if top_k >= self.num_features:
            return torch.ones_like(probs)

        _, primary_indices = torch.topk(probs, k=top_k)
        hard_gates = torch.zeros_like(probs)
        hard_gates.scatter_(0, self.expand_indices(primary_indices), 1.0)
        return hard_gates + probs - probs.detach()

    def selected_indices(self) -> list[int]:
        probs = self.probabilities().detach()
        primary_indices = torch.topk(probs, k=self.top_k()).indices
        return self.expand_indices(primary_indices).cpu().tolist()

    def selected_names(self) -> list[str]:
        return [self.feature_names[i] for i in self.selected_indices()]

    def selection_loss(self) -> torch.Tensor:
        probs = self.probabilities()
        target_active_ratio = probs.new_tensor(self.top_k() / self.num_features)
        active_count_loss = (probs.mean() - target_active_ratio).pow(2)
        entropy = -(
            probs * torch.log(probs.clamp_min(1e-6))
            + (1.0 - probs) * torch.log((1.0 - probs).clamp_min(1e-6))
        ).mean()
        return (
            active_count_loss
            + self.config.generator_feature_entropy_weight * entropy
            + self._dependency_coherence_loss(probs)
        )

    def top_k(self) -> int:
        configured = self.config.generator_backbone_top_k
        if configured is None:
            return self.num_features

        return max(1, min(int(configured), self.num_features))

    def expand_indices(self, indices: torch.Tensor) -> torch.Tensor:
        expanded: set[int] = set()
        for index in indices.detach().cpu().tolist():
            self._add_with_dependencies(int(index), expanded)

        return torch.tensor(
            sorted(expanded),
            device=indices.device,
            dtype=torch.long,
        )

    def _build_dependency_indices(self) -> dict[int, list[int]]:
        name_to_index = {name: i for i, name in enumerate(self.feature_names)}
        dependency_indices: dict[int, list[int]] = {}

        for feature_name, dependencies in self.feature_dependencies.items():
            if feature_name not in name_to_index:
                continue

            child_index = name_to_index[feature_name]
            dependency_indices[child_index] = [
                name_to_index[dependency]
                for dependency in dependencies
                if dependency in name_to_index
            ]

        return dependency_indices

    def _add_with_dependencies(self, index: int, expanded: set[int]):
        if index in expanded:
            return

        expanded.add(index)
        for dependency_index in self._dependency_indices.get(index, []):
            self._add_with_dependencies(dependency_index, expanded)

    def _dependency_coherence_loss(self, probs: torch.Tensor) -> torch.Tensor:
        losses = []
        for child_index, dependency_indices in self._dependency_indices.items():
            child_prob = probs[child_index]
            for dependency_index in dependency_indices:
                losses.append(torch.relu(child_prob - probs[dependency_index]).pow(2))

        if not losses:
            return probs.new_tensor(0.0)

        return torch.stack(losses).mean()
