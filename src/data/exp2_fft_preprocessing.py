"""FFT-augmented preprocessing for Experiment 2.b."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXP2_PREPROC_PATH = _REPO_ROOT / "Exp2" / "Preprocessing"


def _ensure_exp2_on_path() -> None:
    for p in (str(_REPO_ROOT), str(_EXP2_PREPROC_PATH)):
        if p not in sys.path:
            sys.path.insert(0, p)


def apply_exp2_fft_preprocessing(
    frame: pd.DataFrame,
    target_col: str,
    fft_modes: int = 32,
) -> tuple[pd.DataFrame, list[str]]:
    """StandardPreprocessor + low-pass FFT reconstruction appended as extra columns.

    For each numeric feature column, reconstructs a smoothed version using only
    the top `fft_modes` frequency components (rfft → zero high-freq → irfft) and
    appends it as a new column <feature>_fft_smooth. This gives the model both the
    raw normalised signal and its low-frequency envelope.
    """
    _ensure_exp2_on_path()
    from PreprocessingMethods import StandardPreprocessor  # noqa: PLC0415

    feature_df = frame.drop(columns=[target_col]).reset_index()
    preprocessed = StandardPreprocessor().transform(feature_df).reset_index(drop=True)

    target_series = frame[[target_col]].reset_index(drop=True)

    numeric_cols = preprocessed.select_dtypes(include=[np.number]).columns.tolist()
    fft_extras: dict[str, np.ndarray] = {}
    for col in numeric_cols:
        signal = preprocessed[col].to_numpy(dtype=np.float32)
        spectrum = np.fft.rfft(signal)
        modes = min(fft_modes, len(spectrum))
        filtered = np.zeros_like(spectrum)
        filtered[:modes] = spectrum[:modes]
        smoothed = np.fft.irfft(filtered, n=len(signal)).astype(np.float32)
        fft_extras[f"{col}_fft_smooth"] = smoothed

    extras_df = pd.DataFrame(fft_extras)
    result = pd.concat([target_series, preprocessed, extras_df], axis=1)
    result.index = frame.index

    new_feature_cols = [col for col in result.columns if col != target_col]
    return result, new_feature_cols
