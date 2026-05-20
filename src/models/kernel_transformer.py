"""Kernel (Performer FAVOR+) Transformer for Experiment 2.d."""

from __future__ import annotations

import torch
from torch import nn

from src.models.layers import KernelAttention, SinusoidalPositionalEncoding, make_causal_mask
from src.utils.config import ModelConfig


class _KernelDecoderLayer(nn.Module):
    """Decoder layer replacing softmax attention with Performer FAVOR+ linear attention."""

    def __init__(self, d_model: int, nhead: int, feature_dim: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.attn = KernelAttention(d_model, nhead, feature_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class KernelTransformerModel(nn.Module):
    """Performer FAVOR+ decoder-only stack → last token → MLP head."""

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        input_kind: str,
        model_config: ModelConfig,
    ):
        super().__init__()
        if model_config.transformer is None:
            raise ValueError("Transformer configuration required.")

        cfg = model_config.transformer
        feature_dim = cfg.feature_dim if cfg.feature_dim is not None else cfg.d_model
        num_layers = cfg.num_layers

        self.input_proj = nn.Linear(input_dim, cfg.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, dropout=cfg.dropout)

        self.layers = nn.ModuleList([
            _KernelDecoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                feature_dim=feature_dim,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)

        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_hidden_size),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_size, horizon),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B, L, input_dim)
        x = self.pos_enc(self.input_proj(inputs))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x[:, -1, :])
