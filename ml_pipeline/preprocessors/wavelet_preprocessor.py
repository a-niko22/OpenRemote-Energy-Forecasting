"""Wavelet-decomposition preprocessor.

Ported from src/data/preprocessing.py:_wavelet_features_for_window so the
logic is identical to what produced reports/exp1_ab_results_snapshot.md.

Each window is decomposed independently, per-feature, with PyWavelets. Only
the current window's samples are used — no future leakage. Approximation and
detail components are reconstructed back to the original lookback length and
either replace the input features or are concatenated.

Shape contract
--------------
Input  : (n_windows, input_len, n_features)
Output : (n_windows, input_len, n_features_out)

where n_features_out depends on `mode`:
  - "concat":  3 * n_features  (original, approx, detail)
  - "replace": 2 * n_features  (approx, detail)

The number of rows (n_windows) is preserved, which is what the pipeline
contract actually requires. Feature count is allowed to change.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    import pywt
except ModuleNotFoundError:  # graceful fallback if PyWavelets is unavailable
    pywt = None  # type: ignore[assignment]

from interfaces.base_preprocessor import BasePreprocessor


def _match_length(values: np.ndarray, target_length: int) -> np.ndarray:
    if len(values) == target_length:
        return values.astype(np.float32)
    if len(values) > target_length:
        return values[:target_length].astype(np.float32)
    pad_width = target_length - len(values)
    return np.pad(values, (0, pad_width), mode="edge").astype(np.float32)


def _fallback_haar(signal: np.ndarray, level: int) -> Tuple[np.ndarray, np.ndarray]:
    """Minimal Haar wavelet fallback used only when PyWavelets is not installed."""
    if level != 1:
        raise ValueError("Fallback Haar wavelet only supports level=1.")
    original_length = len(signal)
    padded = signal.astype(np.float32)
    if len(padded) % 2 != 0:
        padded = np.concatenate([padded, padded[-1:]], axis=0)
    even = padded[0::2]
    odd = padded[1::2]
    approx_coeff = (even + odd) / np.sqrt(2.0)
    detail_coeff = (even - odd) / np.sqrt(2.0)
    approx_pairs = approx_coeff / np.sqrt(2.0)
    detail_pairs = detail_coeff / np.sqrt(2.0)
    approx_signal = np.empty_like(padded)
    detail_signal = np.empty_like(padded)
    approx_signal[0::2] = approx_pairs
    approx_signal[1::2] = approx_pairs
    detail_signal[0::2] = detail_pairs
    detail_signal[1::2] = -detail_pairs
    return _match_length(approx_signal, original_length), _match_length(detail_signal, original_length)


def _decompose_window(
    window: np.ndarray,
    wavelet_name: str,
    level: int,
    mode: str,
) -> np.ndarray:
    lookback, num_features = window.shape
    approx_features = []
    detail_features = []

    for feature_index in range(num_features):
        signal = window[:, feature_index]
        if pywt is not None:
            max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet_name).dec_len)
            use_level = max(1, min(level, max_level)) if max_level > 0 else 1
            coeffs = pywt.wavedec(signal, wavelet_name, level=use_level, mode="periodization")
            approx_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
            detail_coeffs = [np.zeros_like(coeffs[0])] + [c.copy() for c in coeffs[1:]]
            approx_signal = _match_length(
                pywt.waverec(approx_coeffs, wavelet_name, mode="periodization"), lookback
            )
            detail_signal = _match_length(
                pywt.waverec(detail_coeffs, wavelet_name, mode="periodization"), lookback
            )
        else:
            if wavelet_name not in {"db1", "haar"}:
                raise ValueError(
                    f"PyWavelets is not installed and fallback only supports db1/haar, "
                    f"got {wavelet_name}."
                )
            approx_signal, detail_signal = _fallback_haar(signal, level)

        approx_features.append(approx_signal)
        detail_features.append(detail_signal)

    approx_arr = np.stack(approx_features, axis=1)
    detail_arr = np.stack(detail_features, axis=1)

    if mode == "replace":
        return np.concatenate([approx_arr, detail_arr], axis=1)
    if mode == "concat":
        return np.concatenate([window, approx_arr, detail_arr], axis=1)
    raise ValueError(f"Unsupported wavelet mode: {mode}")


class WaveletPreprocessor(BasePreprocessor):
    """Wavelet decomposition applied independently to each window."""

    def __init__(
        self,
        wavelet_name: str = "db1",
        level: int = 1,
        mode: str = "concat",
    ):
        if mode not in {"concat", "replace"}:
            raise ValueError(f"mode must be 'concat' or 'replace', got {mode!r}")
        self.wavelet_name = wavelet_name
        self.level = int(level)
        self.mode = mode
        self._fitted = False
        self._n_features_in: int | None = None
        self._n_features_out: int | None = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 3:
            raise ValueError(
                "WaveletPreprocessor expects 3D input "
                f"(n_windows, input_len, n_features), got shape {X_arr.shape}."
            )
        self._n_features_in = int(X_arr.shape[-1])
        if self.mode == "concat":
            self._n_features_out = 3 * self._n_features_in
        else:
            self._n_features_out = 2 * self._n_features_in
        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("WaveletPreprocessor must be fit before transform.")
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 3:
            raise ValueError(
                "WaveletPreprocessor expects 3D input, got shape "
                f"{X_arr.shape}."
            )

        if X_arr.shape[0] == 0:
            assert self._n_features_out is not None
            return np.empty((0, X_arr.shape[1], self._n_features_out), dtype=np.float32)

        transformed = [
            _decompose_window(window, self.wavelet_name, self.level, self.mode)
            for window in X_arr
        ]
        return np.stack(transformed).astype(np.float32)

    def get_config(self):
        return {
            "type": "WaveletPreprocessor",
            "wavelet_name": self.wavelet_name,
            "level": self.level,
            "mode": self.mode,
            "n_features_in": self._n_features_in,
            "n_features_out": self._n_features_out,
        }