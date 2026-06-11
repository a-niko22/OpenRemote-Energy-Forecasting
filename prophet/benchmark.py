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


def compute_baseline_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mask = actual != 0
    mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100) if mask.any() else float("nan")
    return {"MAE": round(mae, 6), "RMSE": round(rmse, 6), "MAPE": round(mape, 4)}


def compute_naive_baselines(df: pd.DataFrame, forecast_periods: int) -> dict:
    """Relative benchmark: persistence and seasonal-naive baselines."""
    train = df.iloc[:-forecast_periods]
    test_actual = df.iloc[-forecast_periods:]["y"].to_numpy()

    # Persistence: repeat last training value for all steps
    last_val = float(train["y"].iloc[-1])
    persistence_preds = np.full(forecast_periods, last_val)

    # Seasonal naive: same hour 7 days ago (168 steps); fall back to persistence if not enough data
    seasonal_offset = 24 * 7
    if len(train) >= seasonal_offset + forecast_periods:
        seasonal_preds = train["y"].iloc[-(seasonal_offset + forecast_periods): -seasonal_offset if seasonal_offset > 0 else None].to_numpy()
        if len(seasonal_preds) != forecast_periods:
            seasonal_preds = persistence_preds
    else:
        seasonal_preds = persistence_preds

    return {
        "persistence": compute_baseline_metrics(test_actual, persistence_preds),
        "seasonal_naive_7d": compute_baseline_metrics(test_actual, seasonal_preds),
    }


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

    baselines = compute_naive_baselines(df, forecast_periods)

    return {
        "model": "prophet",
        "timestamp": datetime.now().isoformat(),
        "data_rows": len(df),
        "forecast_horizon_hours": FORECAST_HOURS,
        "held_out_test": held_out,
        "cross_validation": cv_metrics,
        "relative_baselines": baselines,
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
    baseline_flat = {}
    for bname, bmetrics in results.get("relative_baselines", {}).items():
        for k, v in bmetrics.items():
            baseline_flat[f"baseline_{bname}_{k}"] = v
    row = {"model": results["model"], "timestamp": results["timestamp"],
           **{f"heldout_{k}": v for k, v in results["held_out_test"].items()},
           **{f"cv_{k}": v for k, v in results["cross_validation"].items()},
           **baseline_flat}
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
    print("\n--- Relative baselines (held-out window) ---")
    for baseline_name, metrics in results.get("relative_baselines", {}).items():
        print(f"  [{baseline_name}]")
        for k, v in metrics.items():
            print(f"    {k:<13}: {v}")
    prophet_mae = results["held_out_test"]["MAE"]
    for baseline_name, metrics in results.get("relative_baselines", {}).items():
        b_mae = metrics["MAE"]
        if b_mae > 0:
            improvement = (b_mae - prophet_mae) / b_mae * 100
            print(f"  Prophet vs {baseline_name}: {improvement:+.1f}% MAE improvement")
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
