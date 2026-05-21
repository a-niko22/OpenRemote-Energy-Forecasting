"""Tests for the Prophet adapter in Luis's shared ml_pipeline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

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
import models.prophet_model as prophet_module
from models.prophet_model import ProphetPipelineModel


class _DummyProphet:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.train_df: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame):
        self.train_df = train_df.copy()
        return self

    def make_future_dataframe(self, periods: int, freq: str):
        assert self.train_df is not None
        total = len(self.train_df) + periods
        ds = pd.date_range(start=self.train_df["ds"].iloc[0], periods=total, freq=freq)
        return pd.DataFrame({"ds": ds})

    def predict(self, future: pd.DataFrame):
        n = len(future)
        return pd.DataFrame({"ds": future["ds"], "yhat": np.arange(n, dtype=np.float64)})


def _windowed_arrays(
    n_windows: int = 8,
    input_len: int = 6,
    n_features: int = 2,
    horizon: int = 3,
):
    rng = np.random.default_rng(321)
    X = rng.normal(size=(n_windows, input_len, n_features)).astype(np.float32)
    y = rng.normal(size=(n_windows, horizon)).astype(np.float32)
    return X, y


class ProphetPipelineWrapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_prophet = prophet_module._Prophet
        prophet_module._Prophet = _DummyProphet

    def tearDown(self) -> None:
        prophet_module._Prophet = self._original_prophet

    def test_wrapper_inherits_from_base_model(self) -> None:
        model = ProphetPipelineModel()
        self.assertIsInstance(model, BaseModel)

    def test_fit_predict_shapes_on_tiny_windowed_data(self) -> None:
        X, y = _windowed_arrays(n_windows=10, input_len=5, n_features=2, horizon=4)
        model = ProphetPipelineModel(target_feature_index=0, freq="h")

        model.fit(X, y)
        preds = model.predict(X[:6])

        self.assertEqual(preds.shape, (6, 4))
        self.assertTrue(model.get_config()["is_fitted"])

    def test_experiment_runner_runs_prophet_wrapper(self) -> None:
        n_samples = 72
        t = np.arange(n_samples, dtype=np.float32)
        price = 30.0 + np.sin(t / 5.0)
        load = 1000.0 + np.cos(t / 7.0)
        X = np.column_stack([price, load]).astype(np.float32)
        y = price.astype(np.float32)

        experiment = Experiment(
            "Prophet synthetic",
            IdentityPreprocessor(),
            ProphetPipelineModel(target_feature_index=0, freq="h"),
            input_len=8,
            horizon=3,
        )
        runner = ExperimentRunner(input_len=8, horizon=3)
        result = runner.run(
            experiment,
            X_train=X[:45],
            y_train=y[:45],
            X_test=X[57:],
            y_test=y[57:],
            X_val=X[45:57],
            y_val=y[45:57],
        )

        self.assertEqual(result["predictions"].shape, result["y_test"].shape)
        self.assertEqual(result["predictions"].shape[1], 3)
        self.assertEqual(result["model_config"]["type"], "prophet_interface_baseline")

    def test_get_config_contains_expected_fields(self) -> None:
        X, y = _windowed_arrays(n_windows=9, input_len=4, n_features=1, horizon=2)
        model = ProphetPipelineModel(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.25,
            freq="h",
            target_feature_index=0,
        )
        model.fit(X, y)
        config = model.get_config()

        self.assertEqual(config["type"], "prophet_interface_baseline")
        self.assertTrue(config["is_fitted"])
        self.assertEqual(config["input_len"], 4)
        self.assertEqual(config["horizon"], 2)
        self.assertEqual(config["seasonality_mode"], "additive")
        self.assertEqual(config["freq"], "h")
        self.assertEqual(config["target_feature_index"], 0)


if __name__ == "__main__":
    unittest.main()
