"""Simple callback primitives used by the trainer."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EarlyStoppingState:
    """Track the best validation score and patience budget."""

    patience: int
    best_value: float = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement: int = 0

    def step(self, model: torch.nn.Module, value: float) -> bool:
        """Update state and return True if training should stop."""
        if value < self.best_value:
            self.best_value = value
            self.best_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience
