"""Tests for dataset loading and split generation."""

from __future__ import annotations

import unittest
import uuid
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from src.data.dataset import load_time_series_data
from src.data.windowing import build_windowed_splits
from src.utils.config import build_experiment_config

TEST_TMP_ROOT = Path("tests/.tmp")
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def make_test_dir() -> Path:
    """Create a writable test directory inside the repo."""
    path = TEST_TMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def _make_config(data_path: str):
    return build_experiment_config(
        {
            "experiment_name": "test_exp",
            "seed": 42,
            "device": "cpu",
            "data": {
                "path": data_path,
                "timestamp_col": "time",
                "target_col": "price",
                "feature_cols": None,
                "resample_freq": None,
            },
            "window": {"lookback": 8, "horizon": 3, "stride": 1},
            "split": {"train_ratio": 0.75, "val_ratio": 0.10, "test_ratio": 0.15},
            "preprocessing": {
                "norm": {"method": "standard"},
                "wavelet": {"wavelet_name": "db1", "level": 1, "mode": "concat"},
                "patch": {"patch_len": 4, "patch_stride": 2},
            },
            "model": {
                "name": "cnn_bilstm",
                "patch_embed_dim": 16,
                "cnn": {
                    "conv_channels": [8, 8],
                    "kernel_size": 3,
                    "use_pooling": False,
                    "pool_kernel": 2,
                    "activation": "relu",
                    "dropout": 0.1,
                },
                "bilstm": {
                    "hidden_size": 8,
                    "num_layers": 1,
                    "dropout": 0.1,
                    "head_hidden_size": 8,
                },
            },
            "training": {
                "batch_size": 4,
                "epochs": 2,
                "lr": 0.001,
                "weight_decay": 0.0,
                "patience": 2,
                "lr_scheduler_patience": 1,
                "lr_scheduler_factor": 0.5,
                "grad_clip": 1.0,
                "num_workers": 0,
            },
            "outputs": {"root_dir": "outputs"},
        }
    )


class DataPipelineTests(unittest.TestCase):
    """Data pipeline tests."""

    def test_default_dataset_schema_matches_expected(self) -> None:
        loaded = load_time_series_data(_make_config("data/processed/final_dataset_full_clean.csv").data)
        self.assertEqual(loaded.target_col, "price")
        self.assertIn("hour", loaded.feature_cols)
        self.assertIn("total_load", loaded.feature_cols)
        self.assertEqual(loaded.frame.index.name, "time")

    def test_window_splits_keep_target_ranges_disjoint(self) -> None:
        tmp_dir = make_test_dir()
        try:
            rows = 40
            frame = pd.DataFrame(
                {
                    "time": pd.date_range("2025-01-01", periods=rows, freq="h"),
                    "price": np.linspace(10, 20, rows),
                    "load": np.linspace(100, 200, rows),
                }
            )
            path = str(tmp_dir / "toy.csv")
            frame.to_csv(path, index=False)

            config = _make_config(path)
            loaded = load_time_series_data(config.data)
            bundle = build_windowed_splits(loaded.frame, config)

            train_ts = bundle.splits["train"].target_timestamps
            val_ts = bundle.splits["val"].target_timestamps
            test_ts = bundle.splits["test"].target_timestamps

            self.assertGreater(train_ts.size, 0)
            self.assertGreater(val_ts.size, 0)
            self.assertGreater(test_ts.size, 0)
            self.assertLess(train_ts.max(), val_ts.min())
            self.assertLess(val_ts.max(), test_ts.min())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
