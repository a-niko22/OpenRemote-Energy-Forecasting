"""Raw DataFrame preprocessors for Experiment 2.

These preprocessors run before windowing because they need column names and a
real timestamp column. They are separate from the existing window-level
StandardScalerPreprocessor and WaveletPreprocessor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from interfaces.base_preprocessor import BasePreprocessor

try:
    from pywt import dwt_max_level, wavedec, waverec, Wavelet
except (ImportError, ModuleNotFoundError):
    dwt_max_level = None
    wavedec = None
    waverec = None
    Wavelet = None


_TIME_PERIODS = {
    "hour": 24,
    "day_of_week": 7,
    "month": 12,
    "quarter": 4,
    "weekend": 2,
}


@dataclass
class _SequentialScaleStats:
    mean: pd.Series
    std: pd.Series
    min_after_standard: pd.Series
    span_after_standard: pd.Series

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "_SequentialScaleStats":
        mean = frame.mean(axis=0)
        std = frame.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        standardised = (frame - mean) / std
        min_after_standard = standardised.min(axis=0)
        max_after_standard = standardised.max(axis=0)
        span_after_standard = (max_after_standard - min_after_standard).replace(0.0, 1.0).fillna(1.0)
        return cls(
            mean=mean,
            std=std,
            min_after_standard=min_after_standard,
            span_after_standard=span_after_standard,
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        standardised = (frame - self.mean) / self.std
        scaled = (standardised - self.min_after_standard) / self.span_after_standard
        return scaled.astype(np.float32)


@dataclass
class _StandardScaleStats:
    mean: pd.Series
    std: pd.Series

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "_StandardScaleStats":
        mean = frame.mean(axis=0)
        std = frame.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        return cls(mean=mean, std=std)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return ((frame - self.mean) / self.std).astype(np.float32)


def _time_series(frame: pd.DataFrame, time_col: str) -> pd.Series:
    if time_col in frame.columns:
        return pd.to_datetime(frame[time_col])
    if isinstance(frame.index, pd.DatetimeIndex):
        return pd.Series(frame.index, index=frame.index)
    raise ValueError(f"Expected a `{time_col}` column or DatetimeIndex for Exp.2 raw preprocessing.")


def _add_time_features(frame: pd.DataFrame, time_col: str) -> pd.DataFrame:
    result = frame.copy()
    timestamp = _time_series(result, time_col)
    result["day"] = timestamp.dt.day.astype(np.float32)
    raw_features = {
        "hour": timestamp.dt.hour.astype(np.float32),
        "day_of_week": timestamp.dt.dayofweek.astype(np.float32),
        "month": timestamp.dt.month.astype(np.float32),
        "quarter": timestamp.dt.quarter.astype(np.float32),
        "weekend": timestamp.dt.dayofweek.isin([5, 6]).astype(np.float32),
    }

    for feature, values in raw_features.items():
        period = _TIME_PERIODS[feature]
        result[f"{feature}_sin"] = np.sin(2.0 * np.pi * values / period).astype(np.float32)
        result[f"{feature}_cos"] = np.cos(2.0 * np.pi * values / period).astype(np.float32)

    if time_col in result.columns:
        result = result.drop(columns=[time_col])
    return result


def _numeric_base_frame(frame: pd.DataFrame, time_col: str) -> pd.DataFrame:
    with_time_features = _add_time_features(frame, time_col)
    numeric = with_time_features.apply(pd.to_numeric, errors="coerce")
    return numeric.select_dtypes(include=[np.number])


class _RawFramePreprocessor(BasePreprocessor):
    def __init__(self, time_col: str = "time"):
        self.time_col = time_col
        self.columns_: list[str] | None = None
        self.fill_values_: pd.Series | None = None
        self._fitted = False

    def _fit_base(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric = _numeric_base_frame(X, self.time_col)
        self.columns_ = list(numeric.columns)
        fill_values = numeric.median(axis=0).fillna(0.0)
        self.fill_values_ = fill_values
        self._fitted = True
        return numeric.reindex(columns=self.columns_).fillna(fill_values).fillna(0.0)

    def _transform_base(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted or self.columns_ is None or self.fill_values_ is None:
            raise RuntimeError(f"{type(self).__name__} must be fit before transform.")
        numeric = _numeric_base_frame(X, self.time_col)
        numeric = numeric.reindex(columns=self.columns_)
        return numeric.fillna(self.fill_values_).fillna(0.0)


class Exp2StandardPreprocessor(_RawFramePreprocessor):
    """Exp.2 raw standard preprocessor: datetime cycles + StandardScaler + MinMaxScaler."""

    def __init__(self, time_col: str = "time"):
        super().__init__(time_col=time_col)
        self.scaler_: _SequentialScaleStats | None = None

    def fit(self, X: pd.DataFrame, y=None):
        del y
        base = self._fit_base(X)
        self.scaler_ = _SequentialScaleStats.fit(base)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_ is None:
            raise RuntimeError("Exp2StandardPreprocessor must be fit before transform.")
        base = self._transform_base(X)
        return self.scaler_.transform(base)

    def get_config(self):
        return {
            "type": "Exp2StandardPreprocessor",
            "time_col": self.time_col,
            "columns": self.columns_,
        }


class Exp2WaveletPreprocessor(_RawFramePreprocessor):
    """Exp.2 raw DWT preprocessor using sym4 detail features."""

    _dimension_groups = [
        ["temperature_2m", "cloud_cover", "wind_speed_10m", "shortwave_radiation"],
        ["total_load", "generation_forecast", "Price", "High", "Low", "Change %", "price"],
    ]

    def __init__(
        self,
        time_col: str = "time",
        wavelet_name: str = "sym4",
        level: int = 3,
        dimension_groups: Iterable[Iterable[str]] | None = None,
    ):
        super().__init__(time_col=time_col)
        self.wavelet_name = wavelet_name
        self.level = int(level)
        self.dimension_groups = [list(group) for group in (dimension_groups or self._dimension_groups)]
        self.detail_columns_: list[str] | None = None
        self.detail_scaler_: _StandardScaleStats | None = None

    def fit(self, X: pd.DataFrame, y=None):
        del y
        base = self._fit_base(X)
        details = self._build_detail_frame(base)
        self.detail_columns_ = list(details.columns)
        self.detail_scaler_ = _StandardScaleStats.fit(details) if self.detail_columns_ else None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        base = self._transform_base(X).astype(np.float32)
        details = self._build_detail_frame(base)
        if self.detail_columns_:
            details = details.reindex(columns=self.detail_columns_).fillna(0.0)
            if self.detail_scaler_ is not None:
                details = self.detail_scaler_.transform(details)
        else:
            details = pd.DataFrame(index=base.index)
        return pd.concat([base, details], axis=1).astype(np.float32)

    def _build_detail_frame(self, base: pd.DataFrame) -> pd.DataFrame:
        detail_data: dict[str, np.ndarray] = {}
        for group in self.dimension_groups:
            for col in group:
                if col not in base.columns:
                    continue
                signal = np.array(base[col].to_numpy(dtype=np.float32), dtype=np.float32, copy=True)
                for detail_idx, detail_signal in enumerate(self._detail_signals(signal), start=1):
                    detail_data[f"{col}_d{detail_idx}"] = detail_signal.astype(np.float32)
        return pd.DataFrame(detail_data, index=base.index)

    def _detail_signals(self, signal: np.ndarray) -> list[np.ndarray]:
        if wavedec is None or waverec is None or dwt_max_level is None or Wavelet is None:
            return [self._fallback_detail(signal)]

        max_level = dwt_max_level(len(signal), Wavelet(self.wavelet_name).dec_len)
        use_level = max(1, min(self.level, max_level)) if max_level > 0 else 1
        coeffs = wavedec(signal, self.wavelet_name, mode="symmetric", level=use_level)
        details = []
        for idx in range(1, len(coeffs)):
            detail_coeffs = [np.zeros_like(c) for c in coeffs]
            detail_coeffs[idx] = coeffs[idx]
            reconstructed = waverec(detail_coeffs, self.wavelet_name, mode="symmetric")[: len(signal)]
            details.append(reconstructed.astype(np.float32))
        return details

    @staticmethod
    def _fallback_detail(signal: np.ndarray) -> np.ndarray:
        if len(signal) <= 2:
            return np.zeros_like(signal, dtype=np.float32)
        smooth = np.convolve(signal, np.array([0.25, 0.5, 0.25], dtype=np.float32), mode="same")
        return (signal - smooth).astype(np.float32)

    def get_config(self):
        return {
            "type": "Exp2WaveletPreprocessor",
            "time_col": self.time_col,
            "wavelet_name": self.wavelet_name,
            "level": self.level,
            "columns": self.columns_,
            "detail_columns": self.detail_columns_,
        }