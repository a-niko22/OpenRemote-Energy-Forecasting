# data/windowing.py
import numpy as np

def make_windows(X, y, input_len, horizon):
    """
    Convert time-ordered sequences into sliding windows for forecasting.

    Args:
        X: (n_samples, n_features) feature array
        y: (n_samples,) target array
        input_len: lookback window size in time steps
        horizon: forecast window size in time steps

    Returns:
        X_windows: (n_windows, input_len, n_features)
        y_windows: (n_windows, horizon)
    """
    X = np.asarray(X)
    y = np.asarray(y)

    n_samples = len(X)
    n_windows = n_samples - input_len - horizon + 1

    if n_windows <= 0:
        raise ValueError(
            f"Not enough samples. Need at least {input_len + horizon}, got {n_samples}."
        )

    X_windows = np.stack([X[i : i + input_len] for i in range(n_windows)])
    y_windows = np.stack([y[i + input_len : i + input_len + horizon] for i in range(n_windows)])

    return X_windows, y_windows