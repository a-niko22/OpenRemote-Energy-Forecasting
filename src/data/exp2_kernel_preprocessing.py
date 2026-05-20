"""Kernel (RBF Nyström) feature-expansion preprocessing for Experiment 2.d."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXP2_PREPROC_PATH = _REPO_ROOT / "Exp2" / "Preprocessing"


def _ensure_exp2_on_path() -> None:
    for p in (str(_REPO_ROOT), str(_EXP2_PREPROC_PATH)):
        if p not in sys.path:
            sys.path.insert(0, p)


def apply_exp2_kernel_preprocessing(
    frame: pd.DataFrame,
    target_col: str,
    n_components: int = 32,
    gamma: float = 1.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """StandardPreprocessor + RBF Nyström kernel feature expansion.

    Applies StandardPreprocessor for normalisation and cyclical encoding,
    then uses RBFSampler (Nyström approximation of the RBF kernel) on the
    numeric feature columns to expand the feature space. The expanded
    components are appended as additional columns, giving the model explicit
    non-linear kernel features alongside the original ones.
    """
    _ensure_exp2_on_path()
    from PreprocessingMethods import StandardPreprocessor  # noqa: PLC0415

    feature_df = frame.drop(columns=[target_col]).reset_index()
    preprocessed = StandardPreprocessor().transform(feature_df).reset_index(drop=True)

    target_series = frame[[target_col]].reset_index(drop=True)

    numeric_cols = preprocessed.select_dtypes(include=[np.number]).columns.tolist()
    numeric_data = preprocessed[numeric_cols].to_numpy(dtype=np.float32)

    # Scale before RBF mapping (RBFSampler is sensitive to input scale)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_data)

    rbf = RBFSampler(gamma=gamma, n_components=n_components, random_state=random_state)
    kernel_features = rbf.fit_transform(scaled).astype(np.float32)

    kernel_df = pd.DataFrame(
        kernel_features,
        columns=[f"rbf_{i}" for i in range(kernel_features.shape[1])],
    )

    result = pd.concat([target_series, preprocessed, kernel_df], axis=1)
    result.index = frame.index

    new_feature_cols = [col for col in result.columns if col != target_col]
    return result, new_feature_cols
