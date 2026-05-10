"""Minimal Exp.1.c/d model usage with synthetic CPU tensors only.

Full training should still go through the shared experiment application / runner.
"""

from __future__ import annotations

import torch

from src.models.cnn_bilstm_transformer import CNNBiLSTMTransformerModel
from src.models.cnn_transformer import CNNTransformerModel


def main() -> None:
    batch_size = 2
    sequence_length = 48
    input_dim = 12
    horizon = 24

    inputs = torch.randn(batch_size, sequence_length, input_dim)

    cnn_transformer = CNNTransformerModel(input_dim=input_dim, horizon=horizon)
    cnn_bilstm_transformer = CNNBiLSTMTransformerModel(input_dim=input_dim, horizon=horizon)

    cnn_transformer_outputs = cnn_transformer(inputs)
    cnn_bilstm_transformer_outputs = cnn_bilstm_transformer(inputs)

    print(f"Input shape: {tuple(inputs.shape)}")
    print(f"CNNTransformerModel output shape: {tuple(cnn_transformer_outputs.shape)}")
    print(f"CNNBiLSTMTransformerModel output shape: {tuple(cnn_bilstm_transformer_outputs.shape)}")


if __name__ == "__main__":
    main()
