import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

BENCHMARK_DIR = "benchmarks"
FORECAST_HOURS = 48


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["ds"])
    return df[["ds", "y"]].dropna()


def build_model() -> Prophet:
    return Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.1,
    )


def compute_metrics(actual: pd.Series, predicted: pd.Series,
                    lower: pd.Series, upper: pd.Series) -> dict:
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mask = actual != 0
    mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
    coverage = float(np.mean((actual >= lower) & (actual <= upper)) * 100)
    return {"MAE": round(mae, 6), "RMSE": round(rmse, 6),
            "MAPE": round(mape, 4), "coverage_pct": round(coverage, 2)}


def run_benchmark(df: pd.DataFrame) -> dict:
    df = df.sort_values("ds").reset_index(drop=True)

    # Detect frequency from data
    freq = pd.infer_freq(df["ds"].head(20)) or "15min"
    periods_per_hour = pd.tseries.frequencies.to_offset("1h") / pd.tseries.frequencies.to_offset(freq)
    forecast_periods = int(FORECAST_HOURS * periods_per_hour)

    train = df.iloc[:-forecast_periods]
    test = df.iloc[-forecast_periods:]

    model = build_model()
    model.fit(train)

    future = model.make_future_dataframe(periods=forecast_periods, freq=freq)
    forecast = model.predict(future)
    forecast_test = forecast.set_index("ds").loc[test["ds"].values]

    held_out = compute_metrics(
        test["y"].values,
        forecast_test["yhat"].values,
        forecast_test["yhat_lower"].values,
        forecast_test["yhat_upper"].values,
    )

    total_days = (df["ds"].max() - df["ds"].min()).days
    initial_days = max(int(total_days * 0.6), FORECAST_HOURS // 24 + 1)
    period_days = max(int(total_days * 0.1), 1)

    print("Running cross-validation (this may take a minute)...")
    cv = cross_validation(model, initial=f"{initial_days} days", period=f"{period_days} days", horizon=f"{FORECAST_HOURS} hours")
    pm = performance_metrics(cv)
    cv_metrics = {
        "MAE": round(float(pm["mae"].mean()), 6),
        "RMSE": round(float(pm["rmse"].mean()), 6),
        "MAPE": round(float(pm["mape"].mean()) * 100, 4),
        "coverage_pct": round(float(pm["coverage"].mean()) * 100, 2),
    }

    return {
        "model": "prophet",
        "timestamp": datetime.now().isoformat(),
        "data_rows": len(df),
        "forecast_horizon_hours": FORECAST_HOURS,
        "held_out_test": held_out,
        "cross_validation": cv_metrics,
        "prophet_config": {
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": True,
            "seasonality_mode": "multiplicative",
            "changepoint_prior_scale": 0.1,
        },
    }


def save_benchmark(results: dict):
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    path = os.path.join(BENCHMARK_DIR, "prophet.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmark to {path}")

    csv_path = os.path.join(BENCHMARK_DIR, "results.csv")
    row = {"model": results["model"], "timestamp": results["timestamp"],
           **{f"heldout_{k}": v for k, v in results["held_out_test"].items()},
           **{f"cv_{k}": v for k, v in results["cross_validation"].items()}}
    row_df = pd.DataFrame([row])
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        # Replace previous prophet row if exists
        existing = existing[existing["model"] != "prophet"]
        row_df = pd.concat([existing, row_df], ignore_index=True)
    row_df.to_csv(csv_path, index=False)
    print(f"Updated {csv_path}")


def print_summary(results: dict):
    print("\n=== Prophet Benchmark ===")
    print(f"Data rows     : {results['data_rows']}")
    print(f"Horizon       : {results['forecast_horizon_hours']}h")
    print("\n--- Held-out test (last 48h) ---")
    for k, v in results["held_out_test"].items():
        print(f"  {k:<15}: {v}")
    print("\n--- Cross-validation (avg) ---")
    for k, v in results["cross_validation"].items():
        print(f"  {k:<15}: {v}")
    print()


if __name__ == "__main__":
    if "--api" in sys.argv:
        from model import load_data_from_api
        print("Loading data from API...")
        df = load_data_from_api()
        print(f"Loaded {len(df)} rows from API")
    else:
        csv_path = "prophet/mock_tariff.csv"
        if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
            csv_path = sys.argv[1]

        if not os.path.exists(csv_path):
            print(f"No data at {csv_path}. Generate mock data first:")
            print("  python prophet/mock_data.py")
            sys.exit(1)

        df = load_data(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")

    results = run_benchmark(df)
    print_summary(results)
    save_benchmark(results)
