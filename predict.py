"""
Inference API for exp2c Encoder-Decoder Transformer energy price forecaster.
Repo: CitrusBoy/exp2c-energy-forecaster

Quickstart (other project):
    from huggingface_hub import snapshot_download
    import sys, numpy as np

    repo_dir = snapshot_download("CitrusBoy/exp2c-energy-forecaster")
    sys.path.insert(0, repo_dir)
    from predict import load_model, predict

    bundle = load_model(repo_dir)
    # X: np.ndarray shape (168, 22) or (batch, 168, 22) — features in FEATURE_COLS order
    forecast = predict(bundle, X)  # shape (24,) or (batch, 24) — EUR/MWh

FEATURE_COLS order must match exactly (see below).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import torch

REPO_ID = "CitrusBoy/exp2c-energy-forecaster"
INPUT_LEN = 168
HORIZON = 24

FEATURE_COLS = [
    "temperature_2m",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
    "total_load",
    "generation_forecast",
    "Price",
    "Open",
    "High",
    "Low",
    "Change %",
    "day",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "quarter_sin",
    "quarter_cos",
    "weekend_sin",
    "weekend_cos",
]


def _add_src_to_path(repo_dir: str | Path) -> None:
    repo_dir = str(Path(repo_dir).resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)


def load_model(repo_dir: str | Path | None = None) -> dict:
    """Load trained model + scalers.

    Args:
        repo_dir: Local path to a cloned / snapshot-downloaded HF repo.
                  If None, downloads from HF Hub automatically.

    Returns:
        dict with keys: model, target_scaler, feature_scaler, horizon, device
    """
    if repo_dir is None:
        from huggingface_hub import snapshot_download
        repo_dir = snapshot_download(REPO_ID)

    repo_dir = Path(repo_dir)
    _add_src_to_path(repo_dir)

    from src.models.encoder_decoder_transformer import EncoderDecoderTransformerModel
    from src.utils.config import ModelConfig, TransformerConfig

    checkpoint = torch.load(
        repo_dir / "best_model.pt", map_location="cpu", weights_only=False
    )
    scalers = joblib.load(repo_dir / "scalers.joblib")

    cfg = checkpoint["config"]["model"]["transformer"]
    transformer_cfg = TransformerConfig(
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        num_decoder_layers=cfg.get("num_decoder_layers", cfg["num_layers"]),
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        activation=cfg.get("activation", "gelu"),
        pooling=cfg.get("pooling") or cfg.get("pool", "mean"),
        head_hidden_size=cfg["head_hidden_size"],
    )
    model_config = ModelConfig(name="encoder_decoder_transformer", transformer=transformer_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncoderDecoderTransformerModel(
        input_dim=checkpoint["input_dim"],
        horizon=checkpoint["horizon"],
        input_kind=checkpoint.get("input_kind", "sequence"),
        model_config=model_config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return {
        "model": model,
        "target_scaler": scalers["target_scaler"],
        "feature_scaler": scalers["feature_scaler"],
        "horizon": checkpoint["horizon"],
        "device": device,
    }


def predict(bundle: dict, X: np.ndarray) -> np.ndarray:
    """Run inference.

    Args:
        bundle: dict returned by load_model()
        X: np.ndarray shape (INPUT_LEN, n_features) or (batch, INPUT_LEN, n_features).
           Features must be unscaled, ordered as FEATURE_COLS.

    Returns:
        np.ndarray shape (HORIZON,) or (batch, HORIZON) — unscaled price forecasts.
    """
    X = np.asarray(X, dtype=np.float32)
    squeeze = X.ndim == 2
    if squeeze:
        X = X[np.newaxis]  # (1, L, F)
    if X.ndim != 3:
        raise ValueError(f"X must be 2-D or 3-D, got shape {X.shape}")

    B, L, F = X.shape
    fs = bundle["feature_scaler"]
    if fs is not None:
        X_scaled = fs.transform(X.reshape(-1, F)).reshape(B, L, F)
    else:
        X_scaled = X

    device = bundle["device"]
    tensor = torch.from_numpy(X_scaled).to(device)

    with torch.no_grad():
        out = bundle["model"](tensor).cpu().numpy()  # (B, horizon)

    ts = bundle["target_scaler"]
    if ts is not None:
        out = ts.inverse_transform(out)

    return out[0] if squeeze else out
