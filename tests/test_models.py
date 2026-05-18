"""Shape tests for both models under all supported input kinds."""

from __future__ import annotations

import unittest

import torch

from src.models.cnn_bilstm import CNNBiLSTMModel
from src.models.cnn_bilstm_transformer import CNNBiLSTMTransformerModel
from src.models.cnn_transformer import CNNTransformerModel
from src.models.cnn_xlstm import CNNXLSTMModel
from src.utils.config import load_experiment_config


class ModelShapeTests(unittest.TestCase):
    """Model shape tests."""

    def test_cnn_bilstm_sequence_shapes(self) -> None:
        config = load_experiment_config("configs/exp1a_cnn_bilstm.yaml")
        model = CNNBiLSTMModel(input_dim=6, horizon=5, input_kind="sequence", model_config=config.model)
        outputs = model(torch.randn(4, 16, 6))
        self.assertEqual(tuple(outputs.shape), (4, 5))

    def test_cnn_bilstm_patch_shapes(self) -> None:
        config = load_experiment_config("configs/exp1a_cnn_bilstm.yaml")
        model = CNNBiLSTMModel(input_dim=12, horizon=5, input_kind="patch", model_config=config.model)
        outputs = model(torch.randn(4, 7, 12))
        self.assertEqual(tuple(outputs.shape), (4, 5))

    def test_cnn_xlstm_sequence_shapes(self) -> None:
        config = load_experiment_config("configs/exp1b_cnn_xlstm.yaml")
        model = CNNXLSTMModel(input_dim=6, horizon=5, input_kind="sequence", model_config=config.model)
        outputs = model(torch.randn(4, 16, 6))
        self.assertEqual(tuple(outputs.shape), (4, 5))

    def test_cnn_xlstm_patch_shapes(self) -> None:
        config = load_experiment_config("configs/exp1b_cnn_xlstm.yaml")
        model = CNNXLSTMModel(input_dim=12, horizon=5, input_kind="patch", model_config=config.model)
        outputs = model(torch.randn(4, 7, 12))
        self.assertEqual(tuple(outputs.shape), (4, 5))

    def test_cnn_bilstm_transformer_sequence_shapes(self) -> None:
        config = load_experiment_config("configs/exp1c_cnn_bilstm_transformer.yaml")
        model = CNNBiLSTMTransformerModel(input_dim=6, horizon=5, input_kind="sequence", model_config=config.model)
        outputs = model(torch.randn(4, 16, 6))
        self.assertEqual(tuple(outputs.shape), (4, 5))

    def test_cnn_bilstm_transformer_patch_shapes(self) -> None:
        config = load_experiment_config("configs/exp1c_cnn_bilstm_transformer.yaml")
        model = CNNBiLSTMTransformerModel(input_dim=12, horizon=5, input_kind="patch", model_config=config.model)
        outputs = model(torch.randn(4, 7, 12))
        self.assertEqual(tuple(outputs.shape), (4, 5))

    def test_cnn_bilstm_transformer_minimal_constructor_shapes(self) -> None:
        model = CNNBiLSTMTransformerModel(input_dim=12, horizon=5)
        outputs = model(torch.randn(4, 16, 12))
        self.assertEqual(tuple(outputs.shape), (4, 5))

    def test_cnn_transformer_sequence_shapes(self) -> None:
        config = load_experiment_config("configs/exp1d_cnn_transformer.yaml")
        model = CNNTransformerModel(input_dim=6, horizon=5, input_kind="sequence", model_config=config.model)
        outputs = model(torch.randn(4, 16, 6))
        self.assertEqual(tuple(outputs.shape), (4, 5))

    def test_cnn_transformer_patch_shapes(self) -> None:
        config = load_experiment_config("configs/exp1d_cnn_transformer.yaml")
        model = CNNTransformerModel(input_dim=12, horizon=5, input_kind="patch", model_config=config.model)
        outputs = model(torch.randn(4, 7, 12))
        self.assertEqual(tuple(outputs.shape), (4, 5))

    def test_cnn_transformer_minimal_constructor_shapes(self) -> None:
        model = CNNTransformerModel(input_dim=12, horizon=5)
        outputs = model(torch.randn(4, 16, 12))
        self.assertEqual(tuple(outputs.shape), (4, 5))


if __name__ == "__main__":
    unittest.main()
