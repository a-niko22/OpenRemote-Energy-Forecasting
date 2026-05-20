"""Dataset loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.features import resolve_feature_columns
from src.utils.config import DataConfig


@dataclass
class LoadedTimeSeriesData:
    """Container for the resolved dataset and schema."""

    frame: pd.DataFrame
    target_col: str
    feature_cols: list[str]


def load_time_series_data(config: DataConfig) -> LoadedTimeSeriesData:
    """Load a tabular time-series CSV and resolve the schema."""
    data_path = Path(config.path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    frame = pd.read_csv(data_path)
    if config.timestamp_col not in frame.columns:
        raise ValueError(
            f"Timestamp column '{config.timestamp_col}' not present in dataset. "
            f"Columns: {list(frame.columns)}"
        )
    if config.target_col not in frame.columns:
        raise ValueError(
            f"Target column '{config.target_col}' not present in dataset. "
            f"Columns: {list(frame.columns)}"
        )

    frame[config.timestamp_col] = pd.to_datetime(frame[config.timestamp_col], errors="coerce")
    frame = frame.dropna(subset=[config.timestamp_col]).sort_values(config.timestamp_col)
    frame = frame.set_index(config.timestamp_col)

    if config.resample_freq:
        frame = frame.resample(config.resample_freq).mean(numeric_only=True)

    frame = frame.ffill().bfill()
    feature_cols = resolve_feature_columns(frame, config.target_col, config.feature_cols)
    selected_cols = [config.target_col] + feature_cols
    frame = frame[selected_cols].copy()

    if frame.empty:
        raise ValueError("Resolved dataset is empty after loading and preprocessing.")

    return LoadedTimeSeriesData(
        frame=frame,
        target_col=config.target_col,
        feature_cols=feature_cols,
    )
