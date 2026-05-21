"""Preprocessors for Experiment 2 models."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ML_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(ML_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE_ROOT))

try:
    from interfaces.base_preprocessor import BasePreprocessor
except ModuleNotFoundError:
    sys.modules.pop("interfaces", None)
    from interfaces.base_preprocessor import BasePreprocessor


class KernelNormPreprocessor(BasePreprocessor):
    """Z-score normalisation per feature for the Kernel (Performer FAVOR+) Transformer.

    Performer kernel attention approximates softmax via random Fourier features.
    Centred, unit-variance inputs keep the dot-products in a stable range for
    the kernel function, preventing near-zero or exploding feature maps.

    Statistics are fitted across all windows and all time steps so the same
    scale is used consistently at inference.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X, y=None):
        """Compute per-feature mean and std from training windows.

        Args:
            X: shape (n_windows, input_len, n_features)
            y: unused
        """
        X_array = np.asarray(X, dtype=np.float32)
        if X_array.ndim != 3:
            raise ValueError("X must be 3-D: (n_windows, input_len, n_features).")
        flat = X_array.reshape(-1, X_array.shape[-1])  # (n_windows * input_len, n_features)
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0)
        return self

    def transform(self, X):
        """Apply z-score normalisation. Shape preserved."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("KernelNormPreprocessor must be fitted before transform().")
        X_array = np.asarray(X, dtype=np.float32)
        return (X_array - self.mean_) / (self.std_ + self.eps)

    def inverse_transform(self, X):
        """Reverse z-score normalisation."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("KernelNormPreprocessor must be fitted before inverse_transform().")
        return np.asarray(X, dtype=np.float32) * (self.std_ + self.eps) + self.mean_

    def get_config(self) -> dict:
        return {
            "type": "KernelNormPreprocessor",
            "eps": self.eps,
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
        }
