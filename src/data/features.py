"""Feature selection helpers."""

from __future__ import annotations

from typing import List

import pandas as pd


def resolve_feature_columns(
    frame: pd.DataFrame,
    target_col: str,
    configured_feature_cols: List[str] | None = None,
) -> List[str]:
    """Resolve feature columns from config or fallback to all other numeric columns."""
    if configured_feature_cols:
        missing = [col for col in configured_feature_cols if col not in frame.columns]
        if missing:
            raise ValueError(f"Configured feature columns not found: {missing}")
        return configured_feature_cols

    numeric_columns = frame.select_dtypes(include=["number"]).columns.tolist()
    return [column for column in numeric_columns if column != target_col]
