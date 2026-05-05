"""Interchangeable preprocessing strategies for Experiment 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

try:
    import pywt
except ModuleNotFoundError:
    pywt = None

from src.data.windowing import WindowSplit
from src.utils.config import ExperimentConfig


@dataclass
class PreprocessedDataBundle:
    """Prepared windows after an interchangeable preprocessing strategy."""

    splits: Dict[str, WindowSplit]
    input_dim: int
    input_kind: str
    input_feature_names: List[str]


def _match_length(values: np.ndarray, target_length: int) -> np.ndarray:
    if len(values) == target_length:
        return values.astype(np.float32)
    if len(values) > target_length:
        return values[:target_length].astype(np.float32)
    pad_width = target_length - len(values)
    return np.pad(values, (0, pad_width), mode="edge").astype(np.float32)


def _wavelet_features_for_window(
    window: np.ndarray,
    feature_names: List[str],
    wavelet_name: str,
    level: int,
    mode: str,
) -> tuple[np.ndarray, List[str]]:
    lookback, num_features = window.shape
    approx_features = []
    detail_features = []
    output_names: List[str] = []

    for feature_index in range(num_features):
        signal = window[:, feature_index]
        if pywt is not None:
            max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet_name).dec_len)
            use_level = max(1, min(level, max_level)) if max_level > 0 else 1
            coeffs = pywt.wavedec(signal, wavelet_name, level=use_level, mode="periodization")

            approx_coeffs = [coeffs[0]] + [np.zeros_like(component) for component in coeffs[1:]]
            detail_coeffs = [np.zeros_like(coeffs[0])] + [component.copy() for component in coeffs[1:]]

            approx_signal = _match_length(pywt.waverec(approx_coeffs, wavelet_name, mode="periodization"), lookback)
            detail_signal = _match_length(pywt.waverec(detail_coeffs, wavelet_name, mode="periodization"), lookback)
        else:
            if wavelet_name not in {"db1", "haar"}:
                raise ValueError(
                    f"Fallback wavelet implementation only supports 'db1' or 'haar', got {wavelet_name}."
                )
            approx_signal, detail_signal = _fallback_haar_reconstruction(signal, level)

        approx_features.append(approx_signal)
        detail_features.append(detail_signal)

        output_names.extend(
            [
                f"{feature_names[feature_index]}_wavelet_approx",
                f"{feature_names[feature_index]}_wavelet_detail",
            ]
        )

    approx_arr = np.stack(approx_features, axis=1)
    detail_arr = np.stack(detail_features, axis=1)

    if mode == "replace":
        return np.concatenate([approx_arr, detail_arr], axis=1), output_names
    if mode == "concat":
        combined = np.concatenate([window, approx_arr, detail_arr], axis=1)
        raw_names = feature_names.copy()
        return combined, raw_names + output_names
    raise ValueError(f"Unsupported wavelet mode: {mode}")


def _fallback_haar_reconstruction(signal: np.ndarray, level: int) -> tuple[np.ndarray, np.ndarray]:
    """Fallback wavelet reconstruction when PyWavelets is unavailable."""
    if level != 1:
        raise ValueError("Fallback Haar wavelet reconstruction currently supports level=1 only.")

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


def apply_preprocessing(
    splits: Dict[str, WindowSplit],
    input_feature_names: List[str],
    preprocess_name: str,
    config: ExperimentConfig,
) -> PreprocessedDataBundle:
    """Apply one preprocessing strategy across all splits."""
    processed_splits: Dict[str, WindowSplit] = {}
    resolved_feature_names = input_feature_names.copy()
    input_kind = "sequence"

    for split_name, split_data in splits.items():
        inputs = split_data.inputs
        if preprocess_name in {"norm", "exp2_standard", "exp2_fft", "exp2_kernel"}:
            processed_inputs = inputs.copy()
            feature_names = input_feature_names.copy()
        elif preprocess_name == "wavelet":
            transformed_windows = []
            feature_names = resolved_feature_names
            for window in inputs:
                transformed_window, feature_names = _wavelet_features_for_window(
                    window=window,
                    feature_names=input_feature_names,
                    wavelet_name=config.preprocessing.wavelet.wavelet_name,
                    level=config.preprocessing.wavelet.level,
                    mode=config.preprocessing.wavelet.mode,
                )
                transformed_windows.append(transformed_window)
            if transformed_windows:
                processed_inputs = np.stack(transformed_windows).astype(np.float32)
            else:
                feature_dim = len(feature_names)
                processed_inputs = np.empty((0, inputs.shape[1], feature_dim), dtype=np.float32)
            resolved_feature_names = feature_names
        elif preprocess_name == "patch":
            input_kind = "patch"
            patch_len = config.preprocessing.patch.patch_len
            patch_stride = config.preprocessing.patch.patch_stride
            num_samples, lookback, num_features = inputs.shape
            if patch_len > lookback:
                raise ValueError(
                    f"Patch length {patch_len} cannot exceed lookback window {lookback}."
                )
            patch_windows = []
            for window in inputs:
                tokens = []
                for start in range(0, lookback - patch_len + 1, patch_stride):
                    patch = window[start : start + patch_len]
                    tokens.append(patch.reshape(-1))
                if not tokens:
                    raise ValueError("Patch configuration produced zero tokens.")
                patch_windows.append(np.stack(tokens))
            if patch_windows:
                processed_inputs = np.stack(patch_windows).astype(np.float32)
            else:
                num_tokens = ((lookback - patch_len) // patch_stride) + 1
                processed_inputs = np.empty((0, num_tokens, patch_len * num_features), dtype=np.float32)
            resolved_feature_names = [
                f"patch_{index}"
                for index in range(processed_inputs.shape[-1] if processed_inputs.size else patch_len * num_features)
            ]
        else:
            raise ValueError(f"Unsupported preprocessing strategy: {preprocess_name}")

        processed_splits[split_name] = WindowSplit(
            inputs=processed_inputs,
            targets=split_data.targets,
            target_timestamps=split_data.target_timestamps,
        )

    input_dim = 0
    for split_data in processed_splits.values():
        if split_data.inputs.size:
            input_dim = int(split_data.inputs.shape[-1])
            break
    if input_dim == 0:
        raise ValueError("No windows were generated for any split.")

    return PreprocessedDataBundle(
        splits=processed_splits,
        input_dim=input_dim,
        input_kind=input_kind,
        input_feature_names=resolved_feature_names,
    )
