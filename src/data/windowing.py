"""Chronological splits, scaling, and supervised window generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.config import ExperimentConfig


SPLIT_ORDER = ("train", "val", "test")


class ForecastDataset(Dataset):
    """Tensor dataset for direct multi-step forecasting."""

    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        return self.inputs[index], self.targets[index]


@dataclass
class ScalerBundle:
    """Bundle of fitted target and feature scalers."""

    target_scaler: "BaseScaler"
    feature_scaler: "BaseScaler" | None
    scaler_name: str


@dataclass
class WindowSplit:
    """A split of supervised windows and aligned metadata."""

    inputs: np.ndarray
    targets: np.ndarray
    target_timestamps: np.ndarray


@dataclass
class WindowedDataBundle:
    """Prepared window data before strategy-specific preprocessing."""

    splits: Dict[str, WindowSplit]
    scaler_bundle: ScalerBundle
    input_feature_names: List[str]


class BaseScaler:
    """Minimal scaler interface for train-only normalization."""

    def fit(self, values: np.ndarray) -> "BaseScaler":
        raise NotImplementedError

    def transform(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class StandardScaler(BaseScaler):
    """Simple standard scaler implemented with NumPy."""

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "StandardScaler":
        self.mean_ = values.mean(axis=0, keepdims=True)
        self.scale_ = values.std(axis=0, keepdims=True)
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.scale_ is not None
        return (values - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.scale_ is not None
        return (values * self.scale_) + self.mean_


class MinMaxScaler(BaseScaler):
    """Simple min-max scaler implemented with NumPy."""

    def __init__(self):
        self.min_: np.ndarray | None = None
        self.range_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "MinMaxScaler":
        self.min_ = values.min(axis=0, keepdims=True)
        max_values = values.max(axis=0, keepdims=True)
        self.range_ = max_values - self.min_
        self.range_[self.range_ == 0.0] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        assert self.min_ is not None and self.range_ is not None
        return (values - self.min_) / self.range_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        assert self.min_ is not None and self.range_ is not None
        return (values * self.range_) + self.min_


def _build_scaler(name: str) -> BaseScaler:
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unsupported scaler method: {name}")


def build_windowed_splits(frame: pd.DataFrame, config: ExperimentConfig) -> WindowedDataBundle:
    """Scale the panel and create leakage-safe chronological windows."""
    target_col = config.data.target_col
    feature_cols = config.data.feature_cols or [column for column in frame.columns if column != target_col]
    scaler_name = config.preprocessing.norm.method

    train_ratio = config.split.train_ratio
    val_ratio = config.split.val_ratio
    test_ratio = config.split.test_ratio
    ratio_total = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, received {ratio_total:.6f}")

    n_rows = len(frame)
    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)

    train_slice = frame.iloc[:train_end]
    target_scaler = _build_scaler(scaler_name)
    target_scaler.fit(train_slice[[target_col]].values)
    target_scaled = target_scaler.transform(frame[[target_col]].values)

    feature_scaler = None
    if feature_cols:
        feature_scaler = _build_scaler(scaler_name)
        feature_scaler.fit(train_slice[feature_cols].values)
        feature_scaled = feature_scaler.transform(frame[feature_cols].values)
        combined = np.concatenate([target_scaled, feature_scaled], axis=1)
    else:
        feature_scaled = np.empty((n_rows, 0), dtype=np.float64)
        combined = target_scaled

    combined = combined.astype(np.float32)
    timestamps = frame.index.to_numpy()

    lookback = config.window.lookback
    horizon = config.window.horizon
    stride = config.window.stride

    split_records = {name: {"inputs": [], "targets": [], "timestamps": []} for name in SPLIT_ORDER}

    max_start = n_rows - horizon + 1
    for target_start in range(lookback, max_start, stride):
        target_end = target_start + horizon
        input_start = target_start - lookback

        if target_end <= train_end:
            split_name = "train"
        elif target_start >= train_end and target_end <= val_end:
            split_name = "val"
        elif target_start >= val_end and target_end <= n_rows:
            split_name = "test"
        else:
            continue

        split_records[split_name]["inputs"].append(combined[input_start:target_start])
        split_records[split_name]["targets"].append(target_scaled[target_start:target_end, 0])
        split_records[split_name]["timestamps"].append(timestamps[target_start:target_end])

    split_bundle: Dict[str, WindowSplit] = {}
    input_feature_names = [target_col] + feature_cols
    for split_name in SPLIT_ORDER:
        inputs = np.asarray(split_records[split_name]["inputs"], dtype=np.float32)
        targets = np.asarray(split_records[split_name]["targets"], dtype=np.float32)
        target_timestamps = np.asarray(split_records[split_name]["timestamps"], dtype=object)
        if inputs.ndim != 3:
            inputs = np.empty((0, lookback, len(input_feature_names)), dtype=np.float32)
        if targets.ndim != 2:
            targets = np.empty((0, horizon), dtype=np.float32)
        if target_timestamps.ndim != 2:
            target_timestamps = np.empty((0, horizon), dtype=object)
        split_bundle[split_name] = WindowSplit(inputs=inputs, targets=targets, target_timestamps=target_timestamps)

    missing_splits = [split_name for split_name, split_data in split_bundle.items() if split_data.inputs.shape[0] == 0]
    if missing_splits:
        raise ValueError(
            "No supervised windows were created for splits "
            f"{missing_splits}. Increase dataset size or reduce lookback/horizon."
        )

    return WindowedDataBundle(
        splits=split_bundle,
        scaler_bundle=ScalerBundle(
            target_scaler=target_scaler,
            feature_scaler=feature_scaler,
            scaler_name=scaler_name,
        ),
        input_feature_names=input_feature_names,
    )


def build_dataloaders(
    splits: Dict[str, WindowSplit],
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    """Build PyTorch dataloaders for each split."""
    dataloaders = {}
    for split_name, split_data in splits.items():
        dataloaders[split_name] = DataLoader(
            ForecastDataset(split_data.inputs, split_data.targets),
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            drop_last=False,
        )
    return dataloaders
