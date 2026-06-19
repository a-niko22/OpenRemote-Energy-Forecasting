"""
Call the NBEATSx energy-price-forecast service and upload the 48h forecast
to the predictedEnergyPrice attribute in OpenRemote.
"""
import time
import requests
import pandas as pd
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OR_URL = "https://localhost"
REALM = "master"
USERNAME = "admin"
PASSWORD = "secret"
ASSET_ID = "7lTTeGoXyVuq9fvH9sQKLL"
ATTRIBUTE = "predictedEnergyPrice"
CSV_PATH = "data/processed/final_dataset_full_clean.csv"
VERIFY_SSL = False

FEATURE_COLS = [
    "price", "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "temperature_2m", "cloud_cover",
    "wind_speed_10m", "shortwave_radiation", "total_load", "generation_forecast"
]
SEQ_LEN = 168
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


def get_forecast(features):
    resp = requests.post(
        "https://localhost/services/energy-price-forecast/predict",
        json={"features": features},
        verify=VERIFY_SSL,
    )
    resp.raise_for_status()
    return resp.json()


def upload_forecast(token, timestamps_ms, values):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    for ts, val in zip(timestamps_ms, values):
        url = f"{OR_URL}/api/{REALM}/asset/predicted/{ASSET_ID}/{ATTRIBUTE}"
        # Use the predicted datapoints endpoint
        resp = requests.put(
            f"{OR_URL}/api/{REALM}/asset/{ASSET_ID}/attribute/{ATTRIBUTE}/{ts}",
            json=round(float(val), 4),
            headers=headers,
            verify=VERIFY_SSL,
        )
        if not resp.ok:
            print(f"  [WARN] ts={ts}: HTTP {resp.status_code} - {resp.text[:80]}")
    print(f"Uploaded {len(values)} forecast datapoints to '{ATTRIBUTE}'.")


def main():
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Use the last SEQ_LEN rows as input
    last_rows = df[FEATURE_COLS].tail(SEQ_LEN)
    if len(last_rows) < SEQ_LEN:
        print(f"Not enough data: need {SEQ_LEN} rows, got {len(last_rows)}")
        return

    features = last_rows.values.tolist()
    print(f"Using rows from {df['time'].iloc[-SEQ_LEN]} to {df['time'].iloc[-1]}")

    print("Calling NBEATSx forecast service...")
    result = get_forecast(features)
    print(f"Response: {result}")

    forecasts = result.get("forecast", result.get("predictions", result.get("values", None)))
    if forecasts is None:
        print("Could not find forecast values in response. Full response:", result)
        return

    print(f"Got {len(forecasts)} forecast values")
    print(f"First few: {forecasts[:5]}")

    # Generate timestamps starting from last data point + 1 hour
    last_time = df["time"].iloc[-1]
    timestamps = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=HORIZON, freq="1h")
    timestamps_ms = [int(t.timestamp() * 1000) for t in timestamps]
    print(f"Forecast period: {timestamps[0]} to {timestamps[-1]}")

    print("Authenticating with OpenRemote...")
    token = get_token()

    print("Uploading forecast...")
    upload_forecast(token, timestamps_ms, forecasts)


if __name__ == "__main__":
    main()
