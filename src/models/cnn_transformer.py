"""CNN-Transformer model for Experiment 1.d."""

from __future__ import annotations

import torch
from torch import nn

from src.models.layers import (
    ConvTemporalEncoder,
    OptionalProjection,
    PatchInputAdapter,
    SinusoidalPositionalEncoding,
    TransformerEncoderBlock,
)
from src.utils.config import ModelConfig


class CNNTransformerModel(nn.Module):
    """CNN feature extractor followed directly by a Transformer encoder."""

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        input_kind: str,
        model_config: ModelConfig,
    ):
        super().__init__()
        if model_config.transformer is None:
            raise ValueError("Transformer configuration is required for CNNTransformerModel.")

        self.input_adapter = PatchInputAdapter(
            input_dim=input_dim,
            input_kind=input_kind,
            patch_embed_dim=model_config.patch_embed_dim,
        )
        self.encoder = ConvTemporalEncoder(
            input_dim=self.input_adapter.output_dim,
            conv_channels=model_config.cnn.conv_channels,
            kernel_size=model_config.cnn.kernel_size,
            use_pooling=model_config.cnn.use_pooling,
            pool_kernel=model_config.cnn.pool_kernel,
            activation=model_config.cnn.activation,
            dropout=model_config.cnn.dropout,
        )

        transformer_cfg = model_config.transformer
        self.transformer_projection = OptionalProjection(
            input_dim=self.encoder.output_dim,
            output_dim=transformer_cfg.d_model,
        )
        self.position_encoding = SinusoidalPositionalEncoding(
            d_model=transformer_cfg.d_model,
            max_len=transformer_cfg.max_len,
        )
        self.transformer = TransformerEncoderBlock(
            d_model=transformer_cfg.d_model,
            nhead=transformer_cfg.nhead,
            num_layers=transformer_cfg.num_layers,
            dim_feedforward=transformer_cfg.dim_feedforward,
            dropout=transformer_cfg.dropout,
            activation=transformer_cfg.activation,
        )
        self.pooling = transformer_cfg.pooling
        self.head = nn.Sequential(
            nn.Linear(transformer_cfg.d_model, transformer_cfg.head_hidden_size),
            nn.ReLU(),
            nn.Dropout(transformer_cfg.dropout),
            nn.Linear(transformer_cfg.head_hidden_size, horizon),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        adapted = self.input_adapter(inputs)
        encoded = self.encoder(adapted)
        transformer_inputs = self.transformer_projection(encoded)
        transformer_inputs = self.position_encoding(transformer_inputs)
        transformer_outputs = self.transformer(transformer_inputs)

        if self.pooling == "mean":
            representation = transformer_outputs.mean(dim=1)
        elif self.pooling == "last":
            representation = transformer_outputs[:, -1, :]
        else:
            raise ValueError(f"Unsupported transformer pooling strategy: {self.pooling}")
        return self.head(representation)
