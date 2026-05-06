"""Shared layers for the Experiment 1 models."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


def build_activation(name: str) -> nn.Module:
    """Build an activation module from its configured name."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation: {name}")


class PatchInputAdapter(nn.Module):
    """Optional projection used when patch tokens are fed into the model."""

    def __init__(self, input_dim: int, input_kind: str, patch_embed_dim: int):
        super().__init__()
        self.input_kind = input_kind
        if input_kind == "patch":
            self.projection = nn.Linear(input_dim, patch_embed_dim)
            self.output_dim = patch_embed_dim
        else:
            self.projection = nn.Identity()
            self.output_dim = input_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs)


class ConvTemporalEncoder(nn.Module):
    """Conv1d stack over the temporal dimension."""

    def __init__(
        self,
        input_dim: int,
        conv_channels: Iterable[int],
        kernel_size: int,
        use_pooling: bool,
        pool_kernel: int,
        activation: str,
        dropout: float,
    ):
        super().__init__()

        layers = []
        in_channels = input_dim
        for out_channels in conv_channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(build_activation(activation))
            if use_pooling:
                layers.append(nn.MaxPool1d(kernel_size=pool_kernel, stride=1, padding=pool_kernel // 2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.network = nn.Sequential(*layers)
        self.output_dim = in_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Inputs are (batch, time, features). Conv1d expects channels first.
        encoded = self.network(inputs.transpose(1, 2))
        return encoded.transpose(1, 2)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for batch-first sequence tensors."""

    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence_length = inputs.size(1)
        if sequence_length > self.encoding.size(1):
            raise ValueError(
                f"Sequence length {sequence_length} exceeds positional encoding max length {self.encoding.size(1)}."
            )
        return inputs + self.encoding[:, :sequence_length, :].to(dtype=inputs.dtype, device=inputs.device)


class OptionalProjection(nn.Module):
    """Project sequence features only when the source and target dims differ."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs)


class TransformerEncoderBlock(nn.Module):
    """Batch-first Transformer encoder wrapper."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = d_model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)


class XLSTMApproxCell(nn.Module):
    """xLSTM-inspired recurrent cell with stabilized exponential-style gates."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        projection_size: int,
        gate_clamp: float,
        stability_eps: float,
        dropout: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.gate_clamp = gate_clamp
        self.stability_eps = stability_eps

        self.input_proj = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_proj = nn.Linear(projection_size, 4 * hidden_size, bias=False)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, projection_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        inputs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size = inputs.size(0)
        if state is None:
            hidden = inputs.new_zeros(batch_size, self.projection_size)
            memory = inputs.new_zeros(batch_size, self.hidden_size)
        else:
            hidden, memory = state

        gate_inputs = self.input_proj(inputs) + self.hidden_proj(hidden)
        i_gate, f_gate, o_gate, g_gate = gate_inputs.chunk(4, dim=-1)

        i_gate = torch.exp(torch.clamp(i_gate, min=-self.gate_clamp, max=self.gate_clamp))
        f_gate = torch.exp(torch.clamp(f_gate, min=-self.gate_clamp, max=self.gate_clamp))
        normalizer = i_gate + f_gate + self.stability_eps

        candidate = torch.tanh(g_gate)
        updated_memory = ((f_gate / normalizer) * memory) + ((i_gate / normalizer) * candidate)
        updated_memory = self.memory_norm(updated_memory)

        output_gate = torch.sigmoid(o_gate)
        hidden_state = output_gate * torch.tanh(self.output_norm(updated_memory))
        projected_hidden = self.dropout(self.output_projection(hidden_state))
        return projected_hidden, (projected_hidden, updated_memory)


class XLSTMApproxStack(nn.Module):
    """Stacked xLSTM-inspired recurrent block with residual connections."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        projection_size: int,
        gate_clamp: float,
        stability_eps: float,
        dropout: float,
    ):
        super().__init__()
        layers = []
        current_input_size = input_size
        for _ in range(num_layers):
            layers.append(
                XLSTMApproxCell(
                    input_size=current_input_size,
                    hidden_size=hidden_size,
                    projection_size=projection_size,
                    gate_clamp=gate_clamp,
                    stability_eps=stability_eps,
                    dropout=dropout,
                )
            )
            current_input_size = projection_size
        self.layers = nn.ModuleList(layers)
        self.output_dim = projection_size

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        layer_input = inputs
        final_state = None

        for layer in self.layers:
            outputs = []
            state = None
            for timestep in range(layer_input.size(1)):
                step_output, state = layer(layer_input[:, timestep, :], state)
                outputs.append(step_output)
            layer_output = torch.stack(outputs, dim=1)

            if layer_output.shape == layer_input.shape:
                layer_output = layer_output + layer_input

            layer_input = layer_output
            final_state = state

        assert final_state is not None
        final_hidden, _ = final_state
        return layer_input, final_hidden
