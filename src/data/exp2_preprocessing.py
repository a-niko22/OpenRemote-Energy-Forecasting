"""Adapter that integrates Exp2's StandardPreprocessor into the windowing pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXP2_PREPROC_PATH = _REPO_ROOT / "Exp2" / "Preprocessing"


def _ensure_exp2_on_path() -> None:
    # Always move both paths to the front so ml_pipeline/interfaces/ cannot
    # shadow the repo-root interfaces/code/ package.
    # Order: _EXP2_PREPROC_PATH first (for PreprocessingMethods import),
    # _REPO_ROOT second (for interfaces.code import), both before ml_pipeline.
    for p in (str(_REPO_ROOT), str(_EXP2_PREPROC_PATH)):
        if p in sys.path:
            sys.path.remove(p)
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
    # ml_pipeline already imported `interfaces` (its own copy without `code/`),
    # so sys.modules["interfaces"] points to the wrong package. Temporarily evict
    # all interfaces.* entries so Python re-imports from repo root (now first in
    # sys.path), then restore so ml_pipeline code keeps working.
    _saved = {k: sys.modules.pop(k) for k in list(sys.modules)
              if k == "interfaces" or k.startswith("interfaces.")}
    sys.modules.pop("PreprocessingMethods", None)
    try:
        from PreprocessingMethods import StandardPreprocessor  # noqa: PLC0415
    finally:
        # Remove the repo-root interfaces we just imported; restore ml_pipeline's.
        for k in list(sys.modules):
            if k == "interfaces" or k.startswith("interfaces."):
                sys.modules.pop(k)
        sys.modules.update(_saved)

    # Expose DatetimeIndex as a column so CyclicalScalling detects it
    feature_df = frame.drop(columns=[target_col]).reset_index()
    preprocessed = StandardPreprocessor().transform(feature_df)
    preprocessed = preprocessed.reset_index(drop=True)

    target_series = frame[[target_col]].reset_index(drop=True)
    result = pd.concat([target_series, preprocessed], axis=1)
    result.index = frame.index

    new_feature_cols = [col for col in result.columns if col != target_col]
    return result, new_feature_cols
