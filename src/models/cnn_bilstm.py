"""CNN-BiLSTM baseline for Experiment 1.a."""

from __future__ import annotations

import torch
from torch import nn

from src.models.layers import ConvTemporalEncoder, PatchInputAdapter
from src.utils.config import ModelConfig


class CNNBiLSTMModel(nn.Module):
    """CNN feature extractor followed by a BiLSTM forecasting head."""

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        input_kind: str,
        model_config: ModelConfig,
    ):
        super().__init__()
        if model_config.bilstm is None:
            raise ValueError("BiLSTM configuration is required for CNNBiLSTMModel.")

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

        bilstm_cfg = model_config.bilstm
        self.sequence_model = nn.LSTM(
            input_size=self.encoder.output_dim,
            hidden_size=bilstm_cfg.hidden_size,
            num_layers=bilstm_cfg.num_layers,
            dropout=bilstm_cfg.dropout if bilstm_cfg.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(bilstm_cfg.hidden_size * 2, bilstm_cfg.head_hidden_size),
            nn.ReLU(),
            nn.Dropout(bilstm_cfg.dropout),
            nn.Linear(bilstm_cfg.head_hidden_size, horizon),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        adapted = self.input_adapter(inputs)
        encoded = self.encoder(adapted)
        _, (hidden, _) = self.sequence_model(encoded)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        representation = torch.cat([forward_hidden, backward_hidden], dim=-1)
        return self.head(representation)
