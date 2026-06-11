"""Prophet adapter that implements the shared ml_pipeline BaseModel contract."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _fit_predict_batch(args: tuple) -> tuple[int, np.ndarray]:
    """Top-level worker for parallel Prophet fitting (must be picklable).

    Returns (batch_index, predictions) where predictions has shape
    (windows_in_batch, horizon).
    """
    (batch_idx, history, horizon, windows_in_batch,
     yearly_seasonality, weekly_seasonality, daily_seasonality,
     seasonality_mode, changepoint_prior_scale,
     synthetic_start, freq) = args

    for _name in ("cmdstanpy", "prophet", "stan"):
        _lg = logging.getLogger(_name)
        _lg.setLevel(logging.WARNING)
        _lg.propagate = False

    from prophet import Prophet  # noqa: PLC0415
    import pandas as _pd        # noqa: PLC0415
    import numpy as _np         # noqa: PLC0415

    m = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
    )
    ds = _pd.date_range(start=synthetic_start, periods=len(history), freq=freq)
    m.fit(_pd.DataFrame({"ds": ds, "y": history}))

    total_ahead = horizon + windows_in_batch - 1
    future = m.make_future_dataframe(periods=total_ahead, freq=freq)
    forecast = m.predict(future)
    yhat = forecast["yhat"].tail(total_ahead).to_numpy(dtype=_np.float32)

    result = _np.empty((windows_in_batch, horizon), dtype=_np.float32)
    for k in range(windows_in_batch):
        result[k] = yhat[k: k + horizon]
    return batch_idx, result


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
    refit_every: int = 1
    n_jobs: int = 1


class ProphetPipelineModel(BaseModel):
    """Pipeline-compatible Prophet baseline (univariate, no external regressors).

    When rolling_refit=True (default), predict() re-fits Prophet every
    *refit_every* test windows using all actuals up to that window's forecast
    anchor. Predictions for windows within a batch are derived by sliding the
    same forecast forward — matching deep-model evaluation while being much
    faster than per-window refitting.

    Speed knobs
    -----------
    refit_every : int  (default 1 — per-window, slowest/most exact)
        Refit Prophet once per this many windows. refit_every=4 on 15-min data
        = hourly refit, ~4x faster with negligible metric impact.
    n_jobs : int  (default 1)
        Number of parallel worker processes for batch fitting.
        -1 -> use all available CPU cores.

    rolling_refit=False keeps the original single-fit extrapolation for fast
    smoke tests / back-compat.
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
        refit_every: int = 1,
        n_jobs: int = 1,
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
            refit_every=max(1, int(refit_every)),
            n_jobs=int(n_jobs),
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
        """Re-fit Prophet every *refit_every* windows, optionally in parallel.

        For each batch anchor i0, Prophet is fit on history up to that point and
        predicts horizon + batch_size - 1 steps. Predictions for intermediate
        windows are obtained by sliding that forecast forward — no extra fits.
        """
        if self.train_series is None:
            raise RuntimeError("train_series not stored — call fit() first.")

        target_idx = self.settings.target_feature_index
        input_len = self.input_len
        horizon = self.horizon
        refit_every = self.settings.refit_every
        n_jobs = self.settings.n_jobs
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        test_series = np.concatenate(
            [X_array[0, :, target_idx], X_array[1:, -1, target_idx]]
        )

        for _noisy in ("cmdstanpy", "prophet", "stan"):
            _lg = logging.getLogger(_noisy)
            _lg.setLevel(logging.WARNING)
            _lg.propagate = False

        batch_starts = list(range(0, n_windows, refit_every))
        n_batches = len(batch_starts)
        log_every = max(1, n_batches // 20)

        s = self.settings

        if n_jobs > 1:
            worker_args = []
            for batch_idx, i0 in enumerate(batch_starts):
                windows_in_batch = min(refit_every, n_windows - i0)
                history = np.concatenate(
                    [self.train_series, test_series[: input_len + i0]]
                )
                worker_args.append((
                    batch_idx, history, horizon, windows_in_batch,
                    s.yearly_seasonality, s.weekly_seasonality, s.daily_seasonality,
                    s.seasonality_mode, s.changepoint_prior_scale,
                    s.synthetic_start, s.freq,
                ))

        all_preds = np.empty((n_windows, horizon), dtype=np.float32)

        if n_jobs == 1:
            for batch_idx, i0 in enumerate(batch_starts):
                if batch_idx % log_every == 0:
                    print(
                        f"[prophet rolling refit] batch {batch_idx}/{n_batches} "
                        f"(window {i0}/{n_windows})",
                        flush=True,
                    )
                windows_in_batch = min(refit_every, n_windows - i0)
                history = np.concatenate(
                    [self.train_series, test_series[: input_len + i0]]
                )
                ds = pd.date_range(
                    start=s.synthetic_start, periods=len(history), freq=s.freq
                )
                m = self._build_model()
                m.fit(pd.DataFrame({"ds": ds, "y": history}))
                total_ahead = horizon + windows_in_batch - 1
                future = m.make_future_dataframe(periods=total_ahead, freq=s.freq)
                forecast = m.predict(future)
                yhat = forecast["yhat"].tail(total_ahead).to_numpy(dtype=np.float32)
                for k in range(windows_in_batch):
                    all_preds[i0 + k] = yhat[k: k + horizon]
        else:
            print(
                f"[prophet rolling refit] {n_batches} batches across {n_jobs} workers "
                f"(refit_every={refit_every})",
                flush=True,
            )
            completed = 0
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                futures = {pool.submit(_fit_predict_batch, a): a[0] for a in worker_args}
                for fut in as_completed(futures):
                    batch_idx, preds = fut.result()
                    i0 = batch_starts[batch_idx]
                    windows_in_batch = min(refit_every, n_windows - i0)
                    all_preds[i0: i0 + windows_in_batch] = preds
                    completed += 1
                    if completed % log_every == 0:
                        print(
                            f"[prophet rolling refit] {completed}/{n_batches} batches done",
                            flush=True,
                        )

        self._n_refit_windows = n_batches
        print(f"[prophet rolling refit] {n_batches}/{n_batches} batches done.", flush=True)
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
            "refit_every": self.settings.refit_every,
            "n_jobs": self.settings.n_jobs,
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
