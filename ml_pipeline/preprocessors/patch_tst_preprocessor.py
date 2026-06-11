"""PatchTST-style patch tokenization.

Ported from src/data/preprocessing.py (the "patch" branch). Each window of
shape (input_len, n_features) is split into overlapping temporal patches of
length `patch_len` and stride `patch_stride`, each patch flattened into a
single token of dimension patch_len * n_features.

Shape contract
--------------
Input  : (n_windows, input_len, n_features)
Output : (n_windows, n_patches, patch_len * n_features)

n_patches = floor((input_len - patch_len) / patch_stride) + 1

This DOES change axes 1 and 2 of the tensor. Row count (n_windows) is
preserved, which is what the windowing contract actually requires. The four
Exp.1 model adapters handle this through `input_kind="patch"`, which routes
the input through their PatchInputAdapter / nn.Linear projection.

Important
---------
When using this preprocessor, instantiate the model with input_kind="patch":

    Experiment("Exp1.a CNN-BiLSTM + Patch",
               PatchTSTPreprocessor(patch_len=24, patch_stride=12),
               CNNBiLSTMPipelineModel(input_kind="patch"))
"""

from __future__ import annotations

import numpy as np

from interfaces.base_preprocessor import BasePreprocessor


class PatchTSTPreprocessor(BasePreprocessor):
    """Slice each window into flattened overlapping patches."""

    def __init__(self, patch_len: int = 24, patch_stride: int = 12):
        if patch_len <= 0:
            raise ValueError(f"patch_len must be positive, got {patch_len}")
        if patch_stride <= 0:
            raise ValueError(f"patch_stride must be positive, got {patch_stride}")
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self._fitted = False
        self._input_len: int | None = None
        self._n_features_in: int | None = None
        self._n_patches: int | None = None
        self._token_dim: int | None = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 3:
            raise ValueError(
                "PatchTSTPreprocessor expects 3D input "
                f"(n_windows, input_len, n_features), got shape {X_arr.shape}."
            )
        _, input_len, n_features = X_arr.shape
        if self.patch_len > input_len:
            raise ValueError(
                f"patch_len {self.patch_len} cannot exceed input_len {input_len}."
            )
        self._input_len = int(input_len)
        self._n_features_in = int(n_features)
        self._n_patches = ((input_len - self.patch_len) // self.patch_stride) + 1
        self._token_dim = self.patch_len * n_features
        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("PatchTSTPreprocessor must be fit before transform.")
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 3:
            raise ValueError(
                f"PatchTSTPreprocessor expects 3D input, got shape {X_arr.shape}."
            )
        n_windows, input_len, n_features = X_arr.shape

        if input_len != self._input_len or n_features != self._n_features_in:
            raise ValueError(
                f"Shape mismatch with fit-time: "
                f"fit was ({self._input_len}, {self._n_features_in}), "
                f"got ({input_len}, {n_features})."
            )

        assert self._n_patches is not None and self._token_dim is not None
        if n_windows == 0:
            return np.empty((0, self._n_patches, self._token_dim), dtype=np.float32)

        token_dim = self._token_dim
        output = np.empty((n_windows, self._n_patches, token_dim), dtype=np.float32)
        for window_idx in range(n_windows):
            window = X_arr[window_idx]
            for patch_idx, start in enumerate(
                range(0, input_len - self.patch_len + 1, self.patch_stride)
            ):
                output[window_idx, patch_idx] = window[start : start + self.patch_len].reshape(-1)
        return output

    def get_config(self):
        return {
            "type": "PatchTSTPreprocessor",
            "patch_len": self.patch_len,
            "patch_stride": self.patch_stride,
            "input_len": self._input_len,
            "n_features_in": self._n_features_in,
            "n_patches": self._n_patches,
            "token_dim": self._token_dim,
        }