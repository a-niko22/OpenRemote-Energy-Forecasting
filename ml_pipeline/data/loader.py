import pandas as pd
from datasets import load_dataset as hf_load_dataset


def load_dataset(repo_id="CitrusBoy/EnergyPriceForecasting",
                 subset="Gas",
                 split="train",
                 target_col="price"):
    """
    Download a dataset from the Hugging Face Hub and return (X, y) as
    numpy arrays. Results are cached under ~/.cache/huggingface/datasets
    after the first run, so subsequent runs don't re-download.

    Note: `price` is intentionally kept in X. The 168h lookback window
    therefore includes the past 168 prices, which is the strongest signal
    in EPF. This is leakage-safe because make_windows slices X[i:i+168]
    for input and y[i+168:i+168+48] for target — the windows never overlap.
    y is still the price column, so the model is trained to predict future
    prices given past prices + exogenous features.
    """
    ds = hf_load_dataset(repo_id, subset, split=split)
    df = ds.to_pandas()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").set_index("time")
    y = df[target_col].values
    X = df.values
    return X, y


def chronological_split(X, y, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (X[:train_end], y[:train_end],
            X[train_end:val_end], y[train_end:val_end],
            X[val_end:], y[val_end:])