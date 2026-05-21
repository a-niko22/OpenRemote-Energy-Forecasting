"""RBF-kernel feature expansion for windowed pipeline data."""

from __future__ import annotations

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler

from interfaces.base_preprocessor import BasePreprocessor


class KernelFeaturePreprocessor(BasePreprocessor):
    """Append RBF random Fourier features to each window timestep."""

    def __init__(
        self,
        n_components: int = 32,
        gamma: float = 1.0,
        random_state: int = 42,
        mode: str = "concat",
    ):
        if n_components <= 0:
            raise ValueError(f"n_components must be positive, got {n_components}")
        if mode != "concat":
            raise ValueError("KernelFeaturePreprocessor currently supports mode='concat' only.")
        self.n_components = int(n_components)
        self.gamma = float(gamma)
        self.random_state = int(random_state)
        self.mode = mode

        self.scaler_: StandardScaler | None = None
        self.sampler_: RBFSampler | None = None
        self._fitted = False
        self._n_features_in: int | None = None
        self._n_features_out: int | None = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 3:
            raise ValueError(
                "KernelFeaturePreprocessor expects 3D input "
                f"(n_windows, input_len, n_features), got shape {X_arr.shape}."
            )
        n_windows, input_len, n_features = X_arr.shape
        if n_windows == 0 or input_len == 0:
            raise ValueError("KernelFeaturePreprocessor cannot be fit on empty windows.")

        flat = X_arr.reshape(-1, n_features)
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(flat)

        self.sampler_ = RBFSampler(
            gamma=self.gamma,
            n_components=self.n_components,
            random_state=self.random_state,
        )
        self.sampler_.fit(scaled)

        self._n_features_in = int(n_features)
        self._n_features_out = int(n_features + self.n_components)
        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted or self.scaler_ is None or self.sampler_ is None:
            raise RuntimeError("KernelFeaturePreprocessor must be fit before transform.")
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 3:
            raise ValueError(
                "KernelFeaturePreprocessor expects 3D input, got shape "
                f"{X_arr.shape}."
            )

        n_windows, input_len, n_features = X_arr.shape
        if n_features != self._n_features_in:
            raise ValueError(
                f"Shape mismatch with fit-time: fit had {self._n_features_in} features, "
                f"got {n_features}."
            )

        if n_windows == 0:
            assert self._n_features_out is not None
            return np.empty((0, input_len, self._n_features_out), dtype=np.float32)

        flat = X_arr.reshape(-1, n_features)
        scaled = self.scaler_.transform(flat)
        kernel_features = self.sampler_.transform(scaled).astype(np.float32)
        kernel_windows = kernel_features.reshape(n_windows, input_len, self.n_components)
        return np.concatenate([X_arr, kernel_windows], axis=2).astype(np.float32)

    def get_config(self):
        return {
            "type": "KernelFeaturePreprocessor",
            "n_components": self.n_components,
            "gamma": self.gamma,
            "random_state": self.random_state,
            "mode": self.mode,
            "n_features_in": self._n_features_in,
            "n_features_out": self._n_features_out,
        }