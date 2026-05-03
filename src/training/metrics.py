"""Regression metrics in original target units."""

from __future__ import annotations

import numpy as np


def regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE, and sMAPE."""
    predictions = predictions.astype(np.float64)
    targets = targets.astype(np.float64)

    mae = float(np.mean(np.abs(predictions - targets)))
    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))

    mape_mask = np.abs(targets) > 1.0e-6
    if np.any(mape_mask):
        mape = float(np.mean(np.abs((predictions[mape_mask] - targets[mape_mask]) / targets[mape_mask])) * 100.0)
    else:
        mape = 0.0

    denom = np.abs(predictions) + np.abs(targets)
    smape_mask = denom > 1.0e-6
    if np.any(smape_mask):
        smape = float(np.mean((2.0 * np.abs(predictions[smape_mask] - targets[smape_mask])) / denom[smape_mask]) * 100.0)
    else:
        smape = 0.0

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "sMAPE": round(smape, 4),
    }
