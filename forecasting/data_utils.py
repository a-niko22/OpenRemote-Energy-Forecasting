"""
Data loading and preprocessing utilities.

Handles:
  - CSV loading with automatic timestamp detection
  - StandardScaler normalisation (price separately for easy inverse-transform)
  - Sliding-window dataset creation
  - Train / val / test split -> PyTorch DataLoaders
  - Scaler persistence (pickle)
"""

import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# Column names that are commonly used for timestamps
_TIMESTAMP_CANDIDATES = [
    'MTU (UTC)', 'MTU (CET/CEST)',
    'time', 'datetime', 'timestamp',
    'date', 'Date', 'DateTime', 'Timestamp', 'index',
]


# -- Dataset -------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -- Loading -------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file and return a DataFrame indexed by timestamp.

    The function tries to detect a timestamp column automatically (by name or
    by parsing the first column).  The DataFrame is sorted by index before
    being returned.
    """
    df = pd.read_csv(path)

    # Try known timestamp column names first
    for col in _TIMESTAMP_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=False, errors='coerce')
            df = df.dropna(subset=[col]).set_index(col)
            break
    else:
        # Fall back: attempt to parse the first column as datetime
        first = df.columns[0]
        try:
            df[first] = pd.to_datetime(df[first], errors='coerce')
            if df[first].notna().mean() > 0.8:
                df = df.dropna(subset=[first]).set_index(first)
        except Exception:
            pass  # Leave as-is; numeric index is fine

    df = df.sort_index()

    # Drop any non-numeric columns that slipped through
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols]

    return df


# -- Windowing & DataLoaders ---------------------------------------------------

def prepare_loaders(
    df: pd.DataFrame,
    lookback: int = 168,
    horizon: int = 48,
    val_ratio: float = 0.10,
    test_ratio: float = 0.15,
    batch_size: int = 32,
):
    """
    Build train / val / test DataLoaders from a DataFrame.

    Input window  : past `lookback` steps of [price + all other features]
    Target        : next `horizon` steps of price (original scale via scaler)

    Returns
    -------
    loaders       : dict with keys 'train', 'val', 'test'
    price_scaler  : fitted StandardScaler for the price column
    feat_scaler   : fitted StandardScaler for the other feature columns
    n_features    : total number of input channels (1 + n_other_features)
    """
    assert 'price' in df.columns, (
        "Dataset must contain a 'price' column. "
        f"Found: {list(df.columns)}"
    )

    feature_cols = [c for c in df.columns if c != 'price']
    df = df[['price'] + feature_cols].copy()

    # Drop columns where more than 50% of values are missing
    thresh = int(len(df) * 0.5)
    before = set(df.columns)
    df = df.dropna(axis=1, thresh=thresh)
    dropped = before - set(df.columns)
    if dropped:
        print(f"  Dropped sparse columns (>50% NaN): {sorted(dropped)}")

    feature_cols = [c for c in df.columns if c != 'price']

    # Forward-fill then back-fill remaining NaNs in features
    df[feature_cols] = df[feature_cols].ffill().bfill()

    # Only drop rows where the target (price) is missing
    df = df.dropna(subset=['price'])

    # -- Scale ----------------------------------------------------------------
    price_scaler = StandardScaler()
    price_vals = price_scaler.fit_transform(df[['price']].values).astype(np.float32)

    if feature_cols:
        feat_scaler = StandardScaler()
        feat_vals = feat_scaler.fit_transform(df[feature_cols].values).astype(np.float32)
    else:
        feat_scaler = None
        feat_vals = np.empty((len(df), 0), dtype=np.float32)

    combined = np.hstack([price_vals, feat_vals])   # (N, 1 + n_feats)

    # -- Sliding windows -------------------------------------------------------
    X, y = [], []
    for i in range(lookback, len(combined) - horizon + 1):
        X.append(combined[i - lookback: i])         # (lookback, n_features)
        y.append(price_vals[i: i + horizon, 0])     # (horizon,)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # -- Split -----------------------------------------------------------------
    n = len(X)
    test_size  = int(n * test_ratio)
    val_size   = int(n * val_ratio)
    train_size = n - val_size - test_size

    splits = {
        'train': (X[:train_size],                          y[:train_size]),
        'val':   (X[train_size: train_size + val_size],    y[train_size: train_size + val_size]),
        'test':  (X[train_size + val_size:],               y[train_size + val_size:]),
    }

    loaders = {
        split: DataLoader(
            TimeSeriesDataset(*data),
            batch_size=batch_size,
            shuffle=(split == 'train'),
            drop_last=False,
        )
        for split, data in splits.items()
    }

    print(f"  Split sizes  ->  train: {train_size}  val: {val_size}  test: {n - train_size - val_size}")

    return loaders, price_scaler, feat_scaler, X.shape[2]


# -- Scaler persistence --------------------------------------------------------

def save_scalers(price_scaler, feat_scaler, path: str):
    with open(path, 'wb') as f:
        pickle.dump({'price': price_scaler, 'features': feat_scaler}, f)


def load_scalers(path: str):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['price'], d['features']
