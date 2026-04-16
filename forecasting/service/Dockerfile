# Build:  docker build -t energy-price-forecast:latest .
# Run:    docker compose up energy-price-forecast

# --- Build stage: install Python dependencies --------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install CPU-only PyTorch first (avoids pulling the ~2 GB CUDA wheel)
RUN pip install --no-cache-dir \
    torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# Install everything else (torch is already satisfied above)
RUN pip install --no-cache-dir -r requirements.txt \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    pydantic==2.7.1 \
    httpx==0.28.1

# --- Runtime stage -----------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Application source
COPY forecasting/ ./forecasting/
COPY saved_models/nbeatsx.pt ./saved_models/nbeatsx.pt
COPY saved_models/nbeatsx_scalers.pkl ./saved_models/nbeatsx_scalers.pkl
COPY dist/ ./dist/
COPY app.py .

# Persistent data mount point (docker-compose volume: service-ml-forecast-data)
RUN mkdir -p /app/deployment/data/configs

EXPOSE 8000

# Non-root user for safety
RUN useradd -m appuser && chown -R appuser /app
USER appuser

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
