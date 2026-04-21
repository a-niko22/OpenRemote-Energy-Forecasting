"""Tests for interchangeable preprocessing strategies."""

from __future__ import annotations

import unittest

import numpy as np

from src.data.preprocessing import apply_preprocessing
from src.data.windowing import WindowSplit
from src.utils.config import load_experiment_config


def _base_splits() -> dict[str, WindowSplit]:
    rng = np.random.default_rng(123)
    inputs = rng.normal(size=(5, 12, 3)).astype(np.float32)
    targets = rng.normal(size=(5, 4)).astype(np.float32)
    timestamps = np.array(
        [[f"2025-01-01T{step:02d}:00:00" for step in range(4)] for _ in range(5)],
        dtype=object,
    )
    return {
        split_name: WindowSplit(inputs=inputs.copy(), targets=targets.copy(), target_timestamps=timestamps.copy())
        for split_name in ("train", "val", "test")
    }


class PreprocessingTests(unittest.TestCase):
    """Preprocessing strategy tests."""

    def test_norm_preprocessing_preserves_shape(self) -> None:
        config = load_experiment_config("configs/exp1a_cnn_bilstm.yaml")
        bundle = apply_preprocessing(_base_splits(), ["price", "load", "temp"], "norm", config)
        self.assertEqual(bundle.input_kind, "sequence")
        self.assertEqual(bundle.splits["train"].inputs.shape, (5, 12, 3))
        self.assertEqual(bundle.input_dim, 3)

    def test_wavelet_preprocessing_expands_features_in_concat_mode(self) -> None:
        config = load_experiment_config("configs/exp1a_cnn_bilstm.yaml")
        bundle = apply_preprocessing(_base_splits(), ["price", "load", "temp"], "wavelet", config)
        self.assertEqual(bundle.splits["train"].inputs.shape[1], 12)
        self.assertEqual(bundle.splits["train"].inputs.shape[2], 9)
        self.assertEqual(bundle.input_kind, "sequence")

    def test_patch_preprocessing_produces_expected_num_patches(self) -> None:
        config = load_experiment_config("configs/exp1a_cnn_bilstm.yaml")
        config.window.lookback = 12
        config.preprocessing.patch.patch_len = 4
        config.preprocessing.patch.patch_stride = 2
        bundle = apply_preprocessing(_base_splits(), ["price", "load", "temp"], "patch", config)
        self.assertEqual(bundle.input_kind, "patch")
        self.assertEqual(bundle.splits["train"].inputs.shape, (5, 5, 12))


if __name__ == "__main__":
    unittest.main()
