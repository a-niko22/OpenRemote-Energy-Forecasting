"""iTransformer for Experiment 2.b (channels-as-tokens, ICLR 2024)."""

from __future__ import annotations

import torch
from torch import nn

from src.utils.config import ModelConfig


class ITransformerModel(nn.Module):
    """Per-channel time-series embedding → attention over channels → target-channel MLP head.

    Each input feature (variate) becomes one token embedded from its full lookback series.
    Self-attention captures cross-variate dependencies. The target variate (index 0) is
    decoded to produce the horizon-step forecast.
    """

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        input_kind: str,
        model_config: ModelConfig,
    ):
        super().__init__()
        if model_config.transformer is None:
            raise ValueError("Transformer configuration is required for ITransformerModel.")

        cfg = model_config.transformer

        # LazyLinear infers the lookback dimension on first forward pass
        self.channel_proj = nn.LazyLinear(cfg.d_model)
        self.channel_norm = nn.LayerNorm(cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_hidden_size),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_size, horizon),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, lookback, input_dim)
        # Transpose to treat each channel as a token: (batch, input_dim, lookback)
        x = inputs.transpose(1, 2)
        x = self.channel_proj(x)        # (batch, input_dim, d_model)
        x = self.channel_norm(x)
        x = self.encoder(x)             # attention over channels: (batch, input_dim, d_model)
        # Target variate is at index 0 (windowing.py places target first in combined array)
        target_repr = x[:, 0, :]       # (batch, d_model)
        return self.head(target_repr)   # (batch, horizon)
