"""FFT + Decoder-Only Transformer for Experiment 2.b."""

from __future__ import annotations

import torch
from torch import nn

from src.models.layers import FFTBlock, SinusoidalPositionalEncoding, make_causal_mask
from src.utils.config import ModelConfig


class FFTDecoderTransformerModel(nn.Module):
    """FFTBlock → input projection → sinusoidal PE → causal decoder → last token → MLP head."""

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
        fft_modes = cfg.fft_modes if cfg.fft_modes is not None else 32
        num_dec = cfg.num_decoder_layers if cfg.num_decoder_layers is not None else cfg.num_layers

        self.fft = FFTBlock(input_dim=input_dim, fft_modes=fft_modes)
        self.input_proj = nn.Linear(input_dim, cfg.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, dropout=cfg.dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_dec)

        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_hidden_size),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_size, horizon),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B, L, input_dim)
        x = self.fft(inputs)
        x = self.pos_enc(self.input_proj(x))
        mask = make_causal_mask(x.size(1), x.device)
        x = self.decoder(tgt=x, memory=x, tgt_mask=mask, memory_mask=mask)
        return self.head(x[:, -1, :])
