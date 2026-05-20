"""Encoder-Decoder Transformer for Experiment 2.c."""

from __future__ import annotations

import torch
from torch import nn

from src.models.layers import SinusoidalPositionalEncoding
from src.utils.config import ModelConfig


class EncoderDecoderTransformerModel(nn.Module):
    """Encoder consumes lookback; learned query tokens cross-attend to encoder memory → MLP head."""

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
        num_dec = cfg.num_decoder_layers if cfg.num_decoder_layers is not None else cfg.num_layers

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

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_dec)

        # Learned query tokens, one per horizon step
        self.query_tokens = nn.Parameter(torch.randn(1, horizon, cfg.d_model))

        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_hidden_size),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B, L, input_dim)
        B = inputs.size(0)
        memory = self.encoder(self.pos_enc(self.input_proj(inputs)))
        queries = self.query_tokens.expand(B, -1, -1)
        out = self.decoder(tgt=queries, memory=memory)
        return self.head(out).squeeze(-1)  # (B, horizon)
