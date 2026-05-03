"""Vanilla Transformer encoder for Experiment 2.a (time-as-tokens)."""

from __future__ import annotations

import torch
from torch import nn

from src.models.layers import SinusoidalPositionalEncoding
from src.utils.config import ModelConfig


class VanillaTransformerModel(nn.Module):
    """Linear input projection → sinusoidal PE → TransformerEncoder → pool → MLP head."""

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        input_kind: str,
        model_config: ModelConfig,
    ):
        super().__init__()
        if model_config.transformer is None:
            raise ValueError("Transformer configuration is required for VanillaTransformerModel.")

        cfg = model_config.transformer
        self.pool = cfg.pool

        self.input_proj = nn.Linear(input_dim, cfg.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, dropout=cfg.dropout)

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
        # inputs: (batch, seq_len, input_dim)
        x = self.input_proj(inputs)
        x = self.pos_enc(x)
        x = self.encoder(x)
        if self.pool == "last":
            x = x[:, -1, :]
        else:
            x = x.mean(dim=1)
        return self.head(x)
