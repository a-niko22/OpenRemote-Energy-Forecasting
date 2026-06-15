# OpenRemote Energy Forecasting

This repository contains the project work for the OpenRemote energy forecasting project. It brings together documentation, datasets, exploratory data analysis, and forecasting experiments in one place.

The current focus is short-term electricity price forecasting, with the goal of improving prediction accuracy compared to the existing Prophet-based approach.

## Energy Price Forecast Service

This repo includes a local `energy-price-forecast` FastAPI service under `forecasting/service` for OpenRemote deployments. The OpenRemote stack routes it through `/services/energy-price-forecast` via HAProxy, without changing the OpenRemote platform source.

### Model files

Before starting the service, place the trained N-BEATSx artifacts in `saved_models/`:

- `saved_models/nbeatsx.pt`
- `saved_models/nbeatsx_scalers.pkl`

You can generate them with the training pipeline, for example:

```bash
python -m forecasting.pipeline.train --data data/processed/final_dataset_full_raw.csv --model nbeatsx
```

### Build and run locally

Build the service image:

```bash
docker build -f forecasting/service/Dockerfile -t energy-price-forecast:latest .
```

Run the full OpenRemote stack plus the local forecasting service:

```bash
docker compose up --build
```

Useful endpoints once the service is up:

- `http://localhost:8000/health` when accessing the container directly
- `https://localhost/services/energy-price-forecast/health` through the OpenRemote proxy
- `https://localhost/services/energy-price-forecast/predict` through the OpenRemote proxy
