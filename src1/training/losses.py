"""Loss builders."""

from __future__ import annotations

from torch import nn


def build_loss() -> nn.Module:
    """Return the baseline regression loss."""
    return nn.MSELoss()
