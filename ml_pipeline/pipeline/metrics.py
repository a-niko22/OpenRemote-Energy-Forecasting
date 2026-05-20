import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def peak_mae(y_true, y_pred, top_pct=0.10):
    """MAE restricted to the top `top_pct` of true prices.
    Mirrors the project goal of beating Prophet specifically on peaks."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    threshold = np.quantile(y_true, 1 - top_pct)
    mask = y_true >= threshold
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def per_horizon_mae(y_true, y_pred):
    """Per-step MAE across the forecast horizon. Shape (horizon,).
    Useful for checking error growth from step 1 to step 48."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred), axis=0)


def compute_metrics(y_true, y_pred):
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "peak_mae_top10": peak_mae(y_true, y_pred, top_pct=0.10),
        "per_horizon_mae": per_horizon_mae(y_true, y_pred).tolist(),
    }