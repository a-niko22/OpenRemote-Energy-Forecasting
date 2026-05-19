"""Per-feature standard (z-score) scaling, fit on training windows only.

Notes
-----
This is one of the three Exp.1 preprocessors. It corresponds to the "norm"
strategy in the proposal. The numbers in reports/exp1_ab_results_snapshot.md
were produced with standard scaling (fit on the train split before windowing
in src/data/windowing.py). This class fits on the train *windows* — which is
the only thing the BasePreprocessor contract exposes — but is mathematically
equivalent for non-overlapping flattened statistics.

Shape is preserved exactly: (n_windows, input_len, n_features) in,
(n_windows, input_len, n_features) out.
"""

from __future__ import annotations

import numpy as np

from interfaces.base_preprocessor import BasePreprocessor


class StandardScalerPreprocessor(BasePreprocessor):
    """Z-score scaler that fits on flattened train windows."""

    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 3:
            raise ValueError(
                "StandardScalerPreprocessor expects 3D input "
                f"(n_windows, input_len, n_features), got shape {X_arr.shape}."
            )
        # Flatten across (n_windows, input_len), keep per-feature stats.
        flat = X_arr.reshape(-1, X_arr.shape[-1])
        self.mean_ = flat.mean(axis=0, keepdims=True).astype(np.float32)
        scale = flat.std(axis=0, keepdims=True).astype(np.float32)
        # Avoid divide-by-zero for constant features.
        scale[scale < self.eps] = 1.0
        self.scale_ = scale
        return self

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScalerPreprocessor must be fit before transform.")
        X_arr = np.asarray(X, dtype=np.float32)
        return (X_arr - self.mean_) / self.scale_

    def inverse_transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScalerPreprocessor must be fit before inverse_transform.")
        X_arr = np.asarray(X, dtype=np.float32)
        return (X_arr * self.scale_) + self.mean_

    def get_config(self):
        return {
            "type": "StandardScalerPreprocessor",
            "eps": self.eps,
            "n_features": None if self.mean_ is None else int(self.mean_.shape[-1]),
        }