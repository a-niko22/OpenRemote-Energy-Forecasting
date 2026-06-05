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


def naive_baseline_predictions(y, input_len, horizon, seasonal_offset=168):
    """Compute persistence and seasonal-naive-7d predictions over sliding windows.

    Matches the same window indexing as make_windows so baselines align with
    model yw_te exactly.

    Args:
        y: 1-D target array (raw test split, not windowed).
        input_len: lookback size (same value passed to make_windows).
        horizon: forecast horizon (same value passed to make_windows).
        seasonal_offset: steps back for seasonal naive (default 168 = 7 days at hourly).

    Returns:
        persistence_preds: (n_windows, horizon) — last observed value repeated.
        seasonal_preds:    (n_windows, horizon) — values seasonal_offset steps back.
    """
    y = np.asarray(y, dtype=np.float64)
    n_windows = len(y) - input_len - horizon + 1
    if n_windows <= 0:
        raise ValueError(
            f"Not enough samples for baselines. Need at least {input_len + horizon}, got {len(y)}."
        )

    persistence_preds = np.empty((n_windows, horizon), dtype=np.float64)
    seasonal_preds = np.empty((n_windows, horizon), dtype=np.float64)

    for i in range(n_windows):
        last_val = y[i + input_len - 1]
        persistence_preds[i] = last_val

        src_start = i + input_len - seasonal_offset
        if src_start >= 0 and src_start + horizon <= len(y):
            seasonal_preds[i] = y[src_start: src_start + horizon]
        else:
            # Fall back to persistence if not enough history.
            seasonal_preds[i] = last_val

    return persistence_preds, seasonal_preds


def compute_relative_baselines(y, input_len, horizon, seasonal_offset=168):
    """Compute MAE and RMSE for persistence and seasonal-naive-7d baselines.

    Args:
        y: 1-D raw test target array.
        input_len: lookback window size.
        horizon: forecast horizon.
        seasonal_offset: steps back for seasonal naive (default 168 = 7 days hourly).

    Returns:
        dict with keys "persistence" and "seasonal_naive_7d", each containing
        {"mae": float, "rmse": float}.
    """
    y = np.asarray(y, dtype=np.float64)
    n_windows = len(y) - input_len - horizon + 1
    y_true = np.stack([y[i + input_len: i + input_len + horizon] for i in range(n_windows)])

    persistence_preds, seasonal_preds = naive_baseline_predictions(
        y, input_len, horizon, seasonal_offset=seasonal_offset
    )

    return {
        "persistence": {
            "mae": mae(y_true, persistence_preds),
            "rmse": rmse(y_true, persistence_preds),
        },
        "seasonal_naive_7d": {
            "mae": mae(y_true, seasonal_preds),
            "rmse": rmse(y_true, seasonal_preds),
        },
    }