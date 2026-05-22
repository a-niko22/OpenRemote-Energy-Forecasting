"""Prophet adapter that implements the shared ml_pipeline BaseModel contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

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


class ProphetPipelineModel(BaseModel):
    """Pipeline-compatible Prophet baseline (univariate, no external regressors)."""

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
    ) -> None:
        self.settings = _ProphetSettings(
            yearly_seasonality=bool(yearly_seasonality),
            weekly_seasonality=bool(weekly_seasonality),
            daily_seasonality=bool(daily_seasonality),
            seasonality_mode=str(seasonality_mode),
            changepoint_prior_scale=float(changepoint_prior_scale),
            freq=str(freq),
            target_feature_index=int(target_feature_index),
        )
        self.model = None
        self.input_len: int | None = None
        self.horizon: int | None = None
        self.is_fitted = False
        self.train_series_length: int | None = None
        self.validation_series_length: int = 0
        self.fit_series_length: int | None = None
        self.fit_timestamp: str | None = None

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

        self.input_len = int(X_array.shape[1])
        self.horizon = int(y_array.shape[1])
        self.train_series_length = train_series_length
        self.validation_series_length = validation_series_length
        self.fit_series_length = int(len(series))
        self.fit_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted or self.model is None or self.horizon is None or self.input_len is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        X_array = self._ensure_3d_features(X)
        n_windows = int(len(X_array))
        if n_windows == 0:
            return np.empty((0, self.horizon), dtype=np.float32)

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