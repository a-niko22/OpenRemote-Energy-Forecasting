"""
Energy Price Forecast Service – FastAPI inference server with OpenRemote integration.

Environment variables (all optional, with defaults):
  ML_LOG_LEVEL              Log level (default: INFO)
  ML_WEBSERVER_ORIGINS      CORS origins as JSON array (default: ["*"])
  ML_MODEL_NAME             Model checkpoint to load (default: nbeatsx)
  ML_MODEL_DIR              Directory with .pt and .pkl files (default: saved_models)
  ML_HOST                   Bind host (default: 0.0.0.0)
  ML_PORT                   Bind port (default: 8000)
  ML_OR_URL                 OpenRemote Manager URL (default: http://localhost:8080)
  ML_OR_KEYCLOAK_URL        Keycloak URL (default: http://localhost:8080/auth)
  ML_OR_REALM               Realm (default: master)
  ML_OR_SERVICE_USER        Service user name
  ML_OR_SERVICE_USER_SECRET Service user secret
  ML_OR_SERVICE_URL         External URL where this service is reachable
  ML_VERIFY_SSL             Verify SSL certs (default: true)
"""

import asyncio
import json
import logging
import os
import ssl
from contextlib import asynccontextmanager

import httpx
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
logger = logging.getLogger("energy-price-forecast")

# ---------------------------------------------------------------------------
# Global model state (populated during lifespan startup)
# ---------------------------------------------------------------------------
_state: dict = {}


# ---------------------------------------------------------------------------
# OpenRemote service registration
# ---------------------------------------------------------------------------
SERVICE_ID = "energy-price-forecast"
SERVICE_LABEL = "Energy Price Forecast"
SERVICE_ICON = "lightning-bolt"
API_ROOT_PATH = "/services/energy-price-forecast"


def _verify_ssl() -> bool | ssl.SSLContext:
    return os.getenv("ML_VERIFY_SSL", "true").lower() not in ("false", "0", "no")


async def _get_access_token(client: httpx.AsyncClient) -> str | None:
    """Obtain an OAuth2 token from Keycloak using service-user credentials."""
    keycloak_url = os.getenv("ML_OR_KEYCLOAK_URL", "http://localhost:8080/auth")
    realm = os.getenv("ML_OR_REALM", "master")
    user = os.getenv("ML_OR_SERVICE_USER", "")
    secret = os.getenv("ML_OR_SERVICE_USER_SECRET", "")

    if not user or not secret:
        logger.warning("No service-user credentials configured — skipping registration")
        return None

    token_url = f"{keycloak_url}/realms/{realm}/protocol/openid-connect/token"
    try:
        resp = await client.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": user,
                "client_secret": secret,
            },
        )
        resp.raise_for_status()
        return resp.json()["access_token"]
    except Exception:
        logger.exception("Failed to obtain access token from Keycloak")
        return None


async def _register_service(client: httpx.AsyncClient, token: str) -> str | None:
    """Register this service with OpenRemote Manager. Returns instanceId."""
    or_url = os.getenv("ML_OR_URL", "http://localhost:8080")
    service_url = os.getenv("ML_OR_SERVICE_URL", "https://localhost")

    payload = {
        "serviceId": SERVICE_ID,
        "label": SERVICE_LABEL,
        "icon": SERVICE_ICON,
        "homepageUrl": f"{service_url}{API_ROOT_PATH}/ui/{{realm}}",
        "status": "AVAILABLE",
    }

    try:
        resp = await client.post(
            f"{or_url}/api/master/service/global",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        data = resp.json()
        instance_id = data.get("instanceId")
        logger.info("Registered with OpenRemote  serviceId=%s  instanceId=%s", SERVICE_ID, instance_id)
        return instance_id
    except Exception:
        logger.exception("Failed to register service with OpenRemote Manager")
        return None


async def _heartbeat_loop(client: httpx.AsyncClient, instance_id: str):
    """Send heartbeat every 40 seconds to keep the service marked as AVAILABLE."""
    or_url = os.getenv("ML_OR_URL", "http://localhost:8080")

    while True:
        await asyncio.sleep(40)
        try:
            token = await _get_access_token(client)
            if token is None:
                continue
            resp = await client.put(
                f"{or_url}/api/master/service/{SERVICE_ID}/{instance_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code != 204:
                logger.warning("Heartbeat returned %s", resp.status_code)
        except Exception:
            logger.exception("Heartbeat failed")


async def _start_registration():
    """Authenticate, register, and start heartbeat loop."""
    verify = _verify_ssl()
    client = httpx.AsyncClient(verify=verify, timeout=15.0)

    token = await _get_access_token(client)
    if token is None:
        return

    instance_id = await _register_service(client, token)
    if instance_id is None:
        await client.aclose()
        return

    _state["_http_client"] = client
    _state["_heartbeat_task"] = asyncio.create_task(_heartbeat_loop(client, instance_id))


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Load model ---
    model_name = os.getenv("ML_MODEL_NAME", "nbeatsx")
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

    # --- Register with OpenRemote ---
    await _start_registration()

    yield

    # --- Cleanup ---
    task = _state.pop("_heartbeat_task", None)
    if task:
        task.cancel()
    client = _state.pop("_http_client", None)
    if client:
        await client.aclose()
    _state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
origins: list = json.loads(os.getenv("ML_WEBSERVER_ORIGINS", '["*"]'))

app = FastAPI(
    title="Energy Price Forecast Service",
    root_path=API_ROOT_PATH,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# UI (embedded in OpenRemote Manager via iframe)
# ---------------------------------------------------------------------------
@app.get("/ui/{realm}")
def ui_page(realm: str):
    return FileResponse("static/index.html", media_type="text/html")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """
    features: 2-D list of shape [seq_len, n_features].
    Column order must match the training dataset:
      [price, hour, day_of_week, month, is_weekend, hour_sin, hour_cos,
       temperature_2m, cloud_cover, wind_speed_10m, shortwave_radiation,
       total_load, generation_forecast]
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
