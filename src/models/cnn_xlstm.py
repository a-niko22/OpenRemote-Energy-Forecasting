"""CNN-xLSTM baseline for Experiment 1.b."""

from __future__ import annotations

import torch
from torch import nn

from src.models.layers import ConvTemporalEncoder, PatchInputAdapter, XLSTMApproxStack
from src.utils.config import ModelConfig


class CNNXLSTMModel(nn.Module):
    """CNN encoder followed by an xLSTM-inspired recurrent stack."""

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        input_kind: str,
        model_config: ModelConfig,
    ):
        super().__init__()
        if model_config.xlstm is None:
            raise ValueError("xLSTM configuration is required for CNNXLSTMModel.")

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

        xlstm_cfg = model_config.xlstm
        self.sequence_model = XLSTMApproxStack(
            input_size=self.encoder.output_dim,
            hidden_size=xlstm_cfg.hidden_size,
            num_layers=xlstm_cfg.num_layers,
            projection_size=xlstm_cfg.projection_size,
            gate_clamp=xlstm_cfg.gate_clamp,
            stability_eps=xlstm_cfg.stability_eps,
            dropout=xlstm_cfg.dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(self.sequence_model.output_dim, xlstm_cfg.head_hidden_size),
            nn.ReLU(),
            nn.Dropout(xlstm_cfg.dropout),
            nn.Linear(xlstm_cfg.head_hidden_size, horizon),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        adapted = self.input_adapter(inputs)
        encoded = self.encoder(adapted)
        _, final_hidden = self.sequence_model(encoded)
        return self.head(final_hidden)
