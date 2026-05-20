"""Adapter that integrates Exp2's StandardPreprocessor into the windowing pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXP2_PREPROC_PATH = _REPO_ROOT / "Exp2" / "Preprocessing"


def _ensure_exp2_on_path() -> None:
    for p in (str(_REPO_ROOT), str(_EXP2_PREPROC_PATH)):
        if p not in sys.path:
            sys.path.insert(0, p)


def apply_exp2_standard_preprocessing(
    frame: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply StandardPreprocessor to feature columns before windowing.

    Resets the DatetimeIndex to a regular column so CyclicalScalling can find it,
    applies StandardPreprocessor to feature columns only (target excluded),
    then restores the original index.

    Returns the preprocessed frame and the updated feature column list.
    """
    _ensure_exp2_on_path()
    from PreprocessingMethods import StandardPreprocessor  # noqa: PLC0415

    # Expose DatetimeIndex as a column so CyclicalScalling detects it
    feature_df = frame.drop(columns=[target_col]).reset_index()
    preprocessed = StandardPreprocessor().transform(feature_df)
    preprocessed = preprocessed.reset_index(drop=True)

    target_series = frame[[target_col]].reset_index(drop=True)
    result = pd.concat([target_series, preprocessed], axis=1)
    result.index = frame.index

    new_feature_cols = [col for col in result.columns if col != target_col]
    return result, new_feature_cols
