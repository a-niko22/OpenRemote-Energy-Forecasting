"""Tests for Exp.1.a/b adapters in Luis's shared ml_pipeline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ML_PIPELINE_ROOT = ROOT / "ml_pipeline"
if str(ML_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE_ROOT))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from interfaces.base_model import BaseModel
from pipeline.experiment import Experiment
from pipeline.experiment_runner import ExperimentRunner
from test_models_and_preprocessors.identity_preprocessor import IdentityPreprocessor
from models.exp1_ab_models import (
    CNNBiLSTMPipelineModel,
    CNNXLSTMPipelineModel,
)
from src.utils.config import BiLSTMConfig, CNNConfig, ModelConfig, XLSTMConfig


def _tiny_exp1a_config() -> ModelConfig:
    return ModelConfig(
        name="cnn_bilstm",
        patch_embed_dim=4,
        cnn=CNNConfig(
            conv_channels=[4],
            kernel_size=3,
            use_pooling=False,
            pool_kernel=2,
            activation="relu",
            dropout=0.0,
        ),
        bilstm=BiLSTMConfig(
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
            head_hidden_size=4,
        ),
    )


def _tiny_exp1b_config() -> ModelConfig:
    return ModelConfig(
        name="cnn_xlstm",
        patch_embed_dim=4,
        cnn=CNNConfig(
            conv_channels=[4],
            kernel_size=3,
            use_pooling=False,
            pool_kernel=2,
            activation="relu",
            dropout=0.0,
        ),
        xlstm=XLSTMConfig(
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
            projection_size=4,
            gate_clamp=5.0,
            stability_eps=1.0e-6,
            head_hidden_size=4,
        ),
    )


def _windowed_arrays(
    n_windows: int = 8,
    input_len: int = 6,
    n_features: int = 3,
    horizon: int = 2,
):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(n_windows, input_len, n_features)).astype(np.float32)
    y = rng.normal(size=(n_windows, horizon)).astype(np.float32)
    return X, y


class Exp1ABPipelineWrapperTests(unittest.TestCase):
    def test_wrappers_inherit_from_correct_base_model(self) -> None:
        exp1a = CNNBiLSTMPipelineModel(
            model_config=_tiny_exp1a_config(),
            device="cpu",
        )
        exp1b = CNNXLSTMPipelineModel(
            model_config=_tiny_exp1b_config(),
            device="cpu",
        )

        self.assertIsInstance(exp1a, BaseModel)
        self.assertIsInstance(exp1b, BaseModel)

    def test_experiment_accepts_pipeline_wrappers(self) -> None:
        Experiment(
            "Exp1.a wrapper",
            IdentityPreprocessor(),
            CNNBiLSTMPipelineModel(
                model_config=_tiny_exp1a_config(),
                device="cpu",
            ),
        )
        Experiment(
            "Exp1.b wrapper",
            IdentityPreprocessor(),
            CNNXLSTMPipelineModel(
                model_config=_tiny_exp1b_config(),
                device="cpu",
            ),
        )

    def test_cnn_bilstm_fit_predict_tiny_data(self) -> None:
        X, y = _windowed_arrays()
        model = CNNBiLSTMPipelineModel(
            model_config=_tiny_exp1a_config(),
            device="cpu",
        )

        model.fit(X, y, epochs=1, batch_size=4, learning_rate=0.001)
        predictions = model.predict(X)

        self.assertEqual(predictions.shape, y.shape)
        self.assertTrue(model.get_config()["is_fitted"])

    def test_cnn_xlstm_fit_predict_tiny_data(self) -> None:
        X, y = _windowed_arrays()
        model = CNNXLSTMPipelineModel(
            model_config=_tiny_exp1b_config(),
            device="cpu",
        )

        model.fit(X, y, epochs=1, batch_size=4, learning_rate=0.001)
        predictions = model.predict(X)

        self.assertEqual(predictions.shape, y.shape)
        self.assertTrue(model.get_config()["is_fitted"])

    def test_experiment_runner_runs_wrapper_on_synthetic_data(self) -> None:
        n_samples = 24
        X = np.column_stack([
            np.linspace(0.0, 1.0, n_samples),
            np.sin(np.arange(n_samples) / 3.0),
            np.cos(np.arange(n_samples) / 4.0),
        ]).astype(np.float32)
        y = (X[:, 0] + X[:, 1] * 0.2).astype(np.float32)
        experiment = Experiment(
            "Exp1.a synthetic",
            IdentityPreprocessor(),
            CNNBiLSTMPipelineModel(
                model_config=_tiny_exp1a_config(),
                device="cpu",
            ),
            input_len=4,
            horizon=2,
        )

        runner = ExperimentRunner(input_len=4, horizon=2)
        result = runner.run(
            experiment,
            X_train=X[:16],
            y_train=y[:16],
            X_test=X[16:],
            y_test=y[16:],
            epochs=1,
            batch_size=4,
            learning_rate=0.001,
        )

        self.assertEqual(result["predictions"].shape, result["y_test"].shape)
        self.assertEqual(result["predictions"].shape[1], 2)
        self.assertEqual(result["model_config"]["type"], "exp1a_cnn_bilstm")


if __name__ == "__main__":
    unittest.main()
