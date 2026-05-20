"""
FastAPI service wrapping amazon/chronos-bolt-base for 48-hour energy price forecasting.
Run: uvicorn forecast_service:app --port 8001
"""

from contextlib import asynccontextmanager

import numpy as np
import torch
from chronos import ChronosBoltPipeline
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

pipeline: ChronosBoltPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("Loading amazon/chronos-bolt-base …")
    pipeline = ChronosBoltPipeline.from_pretrained(
        "amazon/chronos-bolt-base",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    print("Model ready.")
    yield
    pipeline = None


app = FastAPI(title="Chronos Energy Forecast", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ForecastRequest(BaseModel):
    prices: list[float]

    @field_validator("prices")
    @classmethod
    def check_length(cls, v: list[float]) -> list[float]:
        if len(v) < 24:
            raise ValueError("Need at least 24 price values as context.")
        return v


class HourEntry(BaseModel):
    hour: int
    price: float
    low: float
    high: float


class ForecastResponse(BaseModel):
    forecast: list[HourEntry]
    analysis: str


def _build_analysis(mean: np.ndarray, low: np.ndarray, high: np.ndarray) -> str:
    peak_h = int(np.argmax(mean)) + 1
    trough_h = int(np.argmin(mean)) + 1
    price_range = float(mean.max() - mean.min())
    avg = float(mean.mean())
    direction = "rising" if mean[-12:].mean() > mean[:12].mean() else "falling"
    return (
        f"Forecast peaks at hour {peak_h} ({mean[peak_h - 1]:.1f} €/MWh) "
        f"and troughs at hour {trough_h} ({mean[trough_h - 1]:.1f} €/MWh). "
        f"48-hour range is {price_range:.1f} €/MWh around an average of {avg:.1f} €/MWh. "
        f"Prices are {direction} over the forecast window."
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(req: ForecastRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    context = torch.tensor([req.prices], dtype=torch.float32)  # [1, context_len]

    # predict returns [num_series, num_samples, prediction_length]
    samples_tensor = pipeline.predict(
        context=context,
        prediction_length=48,
        num_samples=20,
    )

    samples = samples_tensor[0].numpy()  # [20, 48]
    mean = samples.mean(axis=0)
    low = np.percentile(samples, 10, axis=0)
    high = np.percentile(samples, 90, axis=0)

    entries = [
        HourEntry(
            hour=i + 1,
            price=round(float(mean[i]), 2),
            low=round(float(low[i]), 2),
            high=round(float(high[i]), 2),
        )
        for i in range(48)
    ]

    return ForecastResponse(
        forecast=entries,
        analysis=_build_analysis(mean, low, high),
    )
