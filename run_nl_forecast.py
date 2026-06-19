"""
Run NL Energy Forecaster (Nazim112/nl-energy-forecaster) from HuggingFace
and upload 48h forecast to OpenRemote nlForecast attribute.
"""
import sys
import requests
import urllib3
import pandas as pd
import numpy as np

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OR_URL = "https://localhost"
REALM = "master"
USERNAME = "admin"
PASSWORD = "secret"
ASSET_ID = "7lTTeGoXyVuq9fvH9sQKLL"
ATTRIBUTE = "nlForecast"
CSV_PATH = "data/processed/final_dataset_full_clean.csv"
VERIFY_SSL = False

FEATURE_COLS = [
    "temperature_2m", "cloud_cover", "wind_speed_10m", "shortwave_radiation",
    "total_load", "generation_forecast",
    "price",  # = Price
    "hour_sin", "hour_cos",
]

INPUT_LEN = 168
HORIZON = 48


def get_token():
    resp = requests.post(
        f"{OR_URL}/auth/realms/{REALM}/protocol/openid-connect/token",
        data={"grant_type": "password", "client_id": "openremote",
              "username": USERNAME, "password": PASSWORD},
        verify=VERIFY_SSL,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def upload(token, timestamps_ms, values):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    ok = 0
    for ts_ms, val in zip(timestamps_ms, values):
        r = requests.put(
            f"{OR_URL}/api/{REALM}/asset/{ASSET_ID}/attribute/{ATTRIBUTE}/{ts_ms}",
            json=round(float(val), 4), headers=headers, verify=VERIFY_SSL,
        )
        if r.ok:
            ok += 1
        else:
            print(f"  [WARN] {ts_ms}: {r.status_code} {r.text[:80]}")
    print(f"Uploaded {ok}/{len(values)} datapoints to '{ATTRIBUTE}'.")


def main():
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    last_time = df["time"].iloc[-1]

    print("Downloading NL Energy Forecaster from HuggingFace...")
    from huggingface_hub import snapshot_download
    repo_dir = snapshot_download("Nazim112/nl-energy-forecaster")
    print(f"Downloaded to {repo_dir}")

    sys.path.insert(0, repo_dir)

    # Use predict.py from repo
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location("predict", os.path.join(repo_dir, "predict.py"))
    predict_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predict_mod)

    print("Loading model...")
    bundle = predict_mod.load_model(repo_dir)

    # Build input using available features, pad missing ones with zeros
    available = [c for c in predict_mod.FEATURE_COLS if c in df.columns or c == "Price"]
    print(f"Model expects {len(predict_mod.FEATURE_COLS)} features: {predict_mod.FEATURE_COLS}")

    # Map CSV columns to model feature columns
    col_map = {
        "Price": "price",
        "temperature_2m": "temperature_2m",
        "cloud_cover": "cloud_cover",
        "wind_speed_10m": "wind_speed_10m",
        "shortwave_radiation": "shortwave_radiation",
        "total_load": "total_load",
        "generation_forecast": "generation_forecast",
        "hour_sin": "hour_sin",
        "hour_cos": "hour_cos",
    }

    last_rows = df.tail(INPUT_LEN).copy()

    # Build feature matrix with zeros for missing columns
    n_features = len(predict_mod.FEATURE_COLS)
    X = np.zeros((INPUT_LEN, n_features), dtype=np.float32)
    for i, feat in enumerate(predict_mod.FEATURE_COLS):
        csv_col = col_map.get(feat)
        if csv_col and csv_col in last_rows.columns:
            X[:, i] = last_rows[csv_col].values
        else:
            # Derive time-based features
            if feat == "day":
                X[:, i] = last_rows["time"].dt.day.values if "time" in last_rows.columns else 0
            elif feat == "day_of_week_sin":
                X[:, i] = np.sin(2 * np.pi * last_rows["day_of_week"].values / 7)
            elif feat == "day_of_week_cos":
                X[:, i] = np.cos(2 * np.pi * last_rows["day_of_week"].values / 7)
            elif feat == "month_sin":
                X[:, i] = np.sin(2 * np.pi * last_rows["month"].values / 12)
            elif feat == "month_cos":
                X[:, i] = np.cos(2 * np.pi * last_rows["month"].values / 12)
            elif feat == "quarter_sin":
                quarter = (last_rows["month"].values - 1) // 3 + 1
                X[:, i] = np.sin(2 * np.pi * quarter / 4)
            elif feat == "quarter_cos":
                quarter = (last_rows["month"].values - 1) // 3 + 1
                X[:, i] = np.cos(2 * np.pi * quarter / 4)
            elif feat == "weekend_sin":
                X[:, i] = np.sin(2 * np.pi * last_rows["is_weekend"].values / 2)
            elif feat == "weekend_cos":
                X[:, i] = np.cos(2 * np.pi * last_rows["is_weekend"].values / 2)
            else:
                print(f"  [INFO] Feature '{feat}' not in CSV, using 0")

    print("Running inference...")
    forecast = predict_mod.predict(bundle, X)
    print(f"Got {len(forecast)} predictions: {forecast[:5]}")

    timestamps = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=HORIZON, freq="1h")
    timestamps_ms = [int(t.timestamp() * 1000) for t in timestamps]
    print(f"Forecast period: {timestamps[0]} to {timestamps[-1]}")

    print("Authenticating...")
    token = get_token()
    upload(token, timestamps_ms, forecast)


if __name__ == "__main__":
    main()
