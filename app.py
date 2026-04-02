"""
ML Forecast Service – FastAPI inference server for energy price forecasting.

Environment variables (all optional, with defaults):
  ML_LOG_LEVEL        Log level (default: INFO)
  ML_WEBSERVER_ORIGINS CORS origins as JSON array (default: ["*"])
  ML_MODEL_NAME       Model checkpoint to load (default: transformer)
  ML_MODEL_DIR        Directory with .pt and .pkl files (default: saved_models)
  ML_HOST             Bind host (default: 0.0.0.0)
  ML_PORT             Bind port (default: 8000)
"""

import json
import logging
import os
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from forecasting.models import build_model
from forecasting.data_utils import load_scalers

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("ML_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("ml-forecast")

# ---------------------------------------------------------------------------
# Global model state (populated during lifespan startup)
# ---------------------------------------------------------------------------
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = os.getenv("ML_MODEL_NAME", "transformer")
    model_dir  = os.getenv("ML_MODEL_DIR",  "saved_models")

    model_path  = os.path.join(model_dir, f"{model_name}.pt")
    scaler_path = os.path.join(model_dir, f"{model_name}_scalers.pkl")

    logger.info("Loading model from %s", model_path)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    model = build_model(
        cfg["model"],
        input_size=cfg["input_size"],
        seq_len=cfg["seq_len"],
        horizon=cfg["horizon"],
        hidden_size=cfg.get("hidden_size", 128),
        num_layers=cfg.get("num_layers", 2),
        d_model=cfg.get("d_model", 128),
        nhead=cfg.get("nhead", 8),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    price_scaler, feat_scaler = load_scalers(scaler_path)

    _state["model"]         = model
    _state["price_scaler"]  = price_scaler
    _state["feat_scaler"]   = feat_scaler
    _state["config"]        = cfg
    logger.info(
        "Model ready  arch=%s  input_size=%d  seq_len=%d  horizon=%d",
        cfg["model"], cfg["input_size"], cfg["seq_len"], cfg["horizon"],
    )
    yield
    _state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
origins: list = json.loads(os.getenv("ML_WEBSERVER_ORIGINS", '["*"]'))

app = FastAPI(title="ML Forecast Service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """
    features: 2-D list of shape [seq_len, n_features].
    Column order must match the training dataset:
      [price, gen_day_ahead, gen_intraday, gen_actual, load_actual,
       load_forecast, temp_2m, cloud_cover, wind_speed_10m,
       shortwave_radiation, hour, day_of_week, month, ...]
    """
    features: list[list[float]]


class PredictResponse(BaseModel):
    predictions: list[float]   # horizon price values in EUR/MWh
    horizon: int
    model: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    cfg = _state["config"]
    return {
        "status": "ok",
        "model":  cfg["model"],
        "horizon": cfg["horizon"],
        "seq_len": cfg["seq_len"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    cfg         = _state["config"]
    model       = _state["model"]
    price_scaler = _state["price_scaler"]
    feat_scaler  = _state["feat_scaler"]

    x = np.array(req.features, dtype=np.float32)   # [seq_len, n_features]

    expected_seq = cfg["seq_len"]
    expected_feat = cfg["input_size"]
    if x.shape != (expected_seq, expected_feat):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Expected features shape ({expected_seq}, {expected_feat}), "
                f"got {tuple(x.shape)}"
            ),
        )

    # Scale — price column is index 0, remaining columns are features
    price_col = x[:, [0]]
    feat_cols = x[:, 1:] if x.shape[1] > 1 else np.empty((x.shape[0], 0), dtype=np.float32)

    price_scaled = price_scaler.transform(price_col)
    feat_scaled  = feat_scaler.transform(feat_cols) if feat_scaler is not None else feat_cols

    x_scaled = np.hstack([price_scaled, feat_scaled]).astype(np.float32)
    tensor = torch.from_numpy(x_scaled).unsqueeze(0)   # [1, seq_len, n_features]

    with torch.no_grad():
        pred_scaled = model(tensor).squeeze(0).numpy()  # [horizon]

    pred_eur_mwh = price_scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).flatten().tolist()

    return PredictResponse(
        predictions=pred_eur_mwh,
        horizon=cfg["horizon"],
        model=cfg["model"],
    )


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=os.getenv("ML_HOST", "0.0.0.0"),
        port=int(os.getenv("ML_PORT", "8000")),
        reload=False,
    )
