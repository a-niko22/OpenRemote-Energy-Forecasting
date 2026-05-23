"""Prophet adapter that implements the shared ml_pipeline BaseModel contract."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

# Suppress Stan/Prophet per-fit noise so ~1000 rolling refits don't flood the console.
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

try:
    from prophet import Prophet as _Prophet
except (ImportError, ModuleNotFoundError):
    _Prophet = None

from interfaces.base_model import BaseModel


@dataclass
class _ProphetSettings:
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = True
    seasonality_mode: str = "multiplicative"
    changepoint_prior_scale: float = 0.1
    freq: str = "h"
    target_feature_index: int = 0
    synthetic_start: str = "2000-01-01 00:00:00"
    rolling_refit: bool = True


class ProphetPipelineModel(BaseModel):
    """Pipeline-compatible Prophet baseline (univariate, no external regressors).

    When rolling_refit=True (default), predict() re-fits Prophet for every test
    window using all actuals up to that window's forecast anchor. This matches
    how deep models are evaluated (each window sees real recent observations as
    input) and makes metrics directly comparable. rolling_refit=False keeps the
    original single-fit extrapolation for fast smoke tests / back-compat.
    """

    model_type = "prophet_interface_baseline"

    def __init__(
        self,
        *,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        seasonality_mode: str = "multiplicative",
        changepoint_prior_scale: float = 0.1,
        freq: str = "h",
        target_feature_index: int = 0,
        rolling_refit: bool = True,
    ) -> None:
        self.settings = _ProphetSettings(
            yearly_seasonality=bool(yearly_seasonality),
            weekly_seasonality=bool(weekly_seasonality),
            daily_seasonality=bool(daily_seasonality),
            seasonality_mode=str(seasonality_mode),
            changepoint_prior_scale=float(changepoint_prior_scale),
            freq=str(freq),
            target_feature_index=int(target_feature_index),
            rolling_refit=bool(rolling_refit),
        )
        self.model = None
        self.train_series: np.ndarray | None = None
        self.input_len: int | None = None
        self.horizon: int | None = None
        self.is_fitted = False
        self.train_series_length: int | None = None
        self.validation_series_length: int = 0
        self.fit_series_length: int | None = None
        self.fit_timestamp: str | None = None
        self._n_refit_windows: int = 0

    @staticmethod
    def _ensure_3d_features(X: Any) -> np.ndarray:
        array = np.asarray(X, dtype=np.float64)
        if array.ndim != 3:
            raise ValueError(
                "X must have shape (n_windows, input_len, n_features). "
                f"Got shape {array.shape}."
            )
        return array

    @staticmethod
    def _ensure_2d_targets(y: Any) -> np.ndarray:
        array = np.asarray(y, dtype=np.float64)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2:
            raise ValueError(
                "y must have shape (n_windows, horizon) or (n_windows,). "
                f"Got shape {array.shape}."
            )
        return array

    @staticmethod
    def _sliding_matrix(sequence: np.ndarray, horizon: int) -> np.ndarray:
        n = len(sequence) - horizon + 1
        if n <= 0:
            raise ValueError("Sequence is too short to build horizon windows.")
        return np.stack([sequence[i: i + horizon] for i in range(n)]).astype(np.float32)

    def _reconstruct_train_series(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of windows.")
        if len(X) == 0:
            raise ValueError("At least one training window is required.")

        target_idx = self.settings.target_feature_index
        n_features = X.shape[2]
        if target_idx < 0 or target_idx >= n_features:
            raise ValueError(
                f"target_feature_index={target_idx} is out of bounds for n_features={n_features}."
            )

        # Reconstruct contiguous target series from overlapping windows:
        # first full history + each next-step target + final horizon tail.
        history = X[0, :, target_idx]
        first_steps = y[:, 0]
        tail = y[-1, 1:] if y.shape[1] > 1 else np.empty((0,), dtype=np.float64)
        return np.concatenate([history, first_steps, tail], axis=0)

    def _build_model(self):
        if _Prophet is None:
            raise ModuleNotFoundError(
                "prophet is required for ProphetPipelineModel. Install with `pip install prophet`."
            )
        return _Prophet(
            yearly_seasonality=self.settings.yearly_seasonality,
            weekly_seasonality=self.settings.weekly_seasonality,
            daily_seasonality=self.settings.daily_seasonality,
            seasonality_mode=self.settings.seasonality_mode,
            changepoint_prior_scale=self.settings.changepoint_prior_scale,
        )

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        epochs=1,
        batch_size=None,
        learning_rate=None,
        **kwargs,
    ):
        del epochs, batch_size, learning_rate, kwargs
        X_array = self._ensure_3d_features(X)
        y_array = self._ensure_2d_targets(y)

        series = self._reconstruct_train_series(X_array, y_array)
        train_series_length = int(len(series))
        validation_series_length = 0
        if X_val is not None or y_val is not None:
            if X_val is None or y_val is None:
                raise ValueError("X_val and y_val must both be provided or both be None.")
            X_val_array = self._ensure_3d_features(X_val)
            y_val_array = self._ensure_2d_targets(y_val)
            validation_series = self._reconstruct_train_series(X_val_array, y_val_array)
            validation_series_length = int(len(validation_series))
            series = np.concatenate([series, validation_series], axis=0)

        ds = pd.date_range(
            start=self.settings.synthetic_start,
            periods=len(series),
            freq=self.settings.freq,
        )
        train_df = pd.DataFrame({"ds": ds, "y": series.astype(np.float64)})

        self.model = self._build_model()
        self.model.fit(train_df)

        # Store the full train+val series so rolling predict() can prepend it.
        self.train_series = series.astype(np.float64)

        self.input_len = int(X_array.shape[1])
        self.horizon = int(y_array.shape[1])
        self.train_series_length = train_series_length
        self.validation_series_length = validation_series_length
        self.fit_series_length = int(len(series))
        self.fit_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted or self.horizon is None or self.input_len is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        X_array = self._ensure_3d_features(X)
        n_windows = int(len(X_array))
        if n_windows == 0:
            return np.empty((0, self.horizon), dtype=np.float32)

        if self.settings.rolling_refit:
            return self._predict_rolling(X_array, n_windows)
        return self._predict_single(n_windows)

    def _predict_rolling(self, X_array: np.ndarray, n_windows: int) -> np.ndarray:
        """Re-fit Prophet for every window, conditioning on all actuals up to
        that window's forecast anchor. Matches the deep-model evaluation protocol
        where each window sees real recent observations as input context."""
        if self.train_series is None:
            raise RuntimeError("train_series not stored — call fit() first.")

        target_idx = self.settings.target_feature_index
        input_len = self.input_len
        horizon = self.horizon

        # Reconstruct contiguous test series from overlapping windows.
        # window i input = test[i : i+input_len], so:
        # test[0:input_len] = X[0, :, idx]
        # test[input_len + j] = X[j+1, -1, idx]  for j in 0..n_windows-2
        test_series = np.concatenate(
            [X_array[0, :, target_idx], X_array[1:, -1, target_idx]]
        )  # length = input_len + n_windows - 1

        # Re-apply suppression here: cmdstanpy/prophet configure their loggers
        # lazily on first use, after module-level setLevel has already run.
        for _noisy in ("cmdstanpy", "prophet", "stan"):
            _lg = logging.getLogger(_noisy)
            _lg.setLevel(logging.WARNING)
            _lg.propagate = False

        all_preds = np.empty((n_windows, horizon), dtype=np.float32)
        log_every = max(1, n_windows // 20)

        for i in range(n_windows):
            if i % log_every == 0:
                print(f"[prophet rolling refit] {i}/{n_windows}", flush=True)

            # History = full train+val series + test actuals up to forecast anchor.
            history = np.concatenate(
                [self.train_series, test_series[: input_len + i]]
            )
            ds = pd.date_range(
                start=self.settings.synthetic_start,
                periods=len(history),
                freq=self.settings.freq,
            )
            train_df = pd.DataFrame({"ds": ds, "y": history})
            m = self._build_model()
            m.fit(train_df)

            future = m.make_future_dataframe(periods=horizon, freq=self.settings.freq)
            forecast = m.predict(future)
            all_preds[i] = forecast["yhat"].tail(horizon).to_numpy(dtype=np.float32)

        self._n_refit_windows = n_windows
        print(f"[prophet rolling refit] {n_windows}/{n_windows} done.", flush=True)
        return all_preds

    def _predict_single(self, n_windows: int) -> np.ndarray:
        """Original single-fit path: one long extrapolation, slice into windows.
        Fast but ignores actual test observations — use only for smoke tests."""
        if self.model is None:
            raise RuntimeError("No fitted model — call fit() first.")
        periods = self.input_len + n_windows + self.horizon - 1
        future = self.model.make_future_dataframe(periods=periods, freq=self.settings.freq)
        forecast = self.model.predict(future)
        future_yhat = np.asarray(forecast["yhat"].tail(periods), dtype=np.float64)
        yhat = future_yhat[self.input_len:]
        return self._sliding_matrix(yhat, self.horizon)

    def get_config(self):
        return {
            "type": self.model_type,
            "is_fitted": self.is_fitted,
            "rolling_refit": self.settings.rolling_refit,
            "n_refit_windows": self._n_refit_windows,
            "input_len": self.input_len,
            "horizon": self.horizon,
            "train_series_length": self.train_series_length,
            "validation_series_length": self.validation_series_length,
            "fit_series_length": self.fit_series_length,
            "fit_timestamp": self.fit_timestamp,
            "yearly_seasonality": self.settings.yearly_seasonality,
            "weekly_seasonality": self.settings.weekly_seasonality,
            "daily_seasonality": self.settings.daily_seasonality,
            "seasonality_mode": self.settings.seasonality_mode,
            "changepoint_prior_scale": self.settings.changepoint_prior_scale,
            "freq": self.settings.freq,
            "target_feature_index": self.settings.target_feature_index,
        }
