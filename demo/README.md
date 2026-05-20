# Energy Price Forecasting Dashboard

Single-file HTML dashboard with a Python backend serving real zero-shot forecasts from [amazon/chronos-bolt-base](https://huggingface.co/amazon/chronos-bolt-base).

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> `chronos-forecasting` pulls in PyTorch. First run also downloads the ~300 MB Chronos-Bolt model weights from HuggingFace.

### 2. Start the forecast service

```bash
uvicorn forecast_service:app --port 8001
```

Wait for `Model ready.` before opening the dashboard. The service exposes:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check |
| `/forecast` | POST | 48-hour probabilistic forecast |

### 3. Open the dashboard

Open `energy_price_forecasting_dashboard.html` in a browser (no build step needed).

The dashboard fetches real Dutch EPEX prices from HuggingFace, then POSTs the last 168 hours to the local service and renders the Chronos-Bolt forecast with p10–p90 confidence band.

## Architecture

```
Browser (HTML file)
  └─ fetch → datasets-server.huggingface.co   (historical prices)
  └─ POST  → localhost:8001/forecast           (Chronos-Bolt predictions)
       └─ forecast_service.py
            └─ ChronosBoltPipeline (amazon/chronos-bolt-base)
```
