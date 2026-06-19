"""
Upload historical energy price data (2024-2025) to OpenRemote asset attributes.

Usage:
    python upload_to_openremote.py

Requires:
    pip install requests pandas
"""

import time
import requests
import pandas as pd

# --- Configuration ---
OR_URL = "https://localhost"
REALM = "master"
USERNAME = "admin"
PASSWORD = "secret"  # change if you set a different admin password
ASSET_ID = "7lTTeGoXyVuq9fvH9sQKLL"
ATTRIBUTE = "pricecPerMWh"
CSV_PATH = "data/processed/final_dataset_full_clean.csv"
START_DATE = "2024-01-01"

VERIFY_SSL = False  # self-signed cert on localhost


def get_token() -> str:
    resp = requests.post(
        f"{OR_URL}/auth/realms/{REALM}/protocol/openid-connect/token",
        data={
            "grant_type": "password",
            "client_id": "openremote",
            "username": USERNAME,
            "password": PASSWORD,
        },
        verify=VERIFY_SSL,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def upload(token: str, df: pd.DataFrame) -> None:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        timestamp_ms = int(row["time"].timestamp() * 1000)
        value = round(float(row["price"]), 4)

        url = f"{OR_URL}/api/{REALM}/asset/{ASSET_ID}/attribute/{ATTRIBUTE}/{timestamp_ms}"
        resp = requests.put(url, json=value, headers=headers, verify=VERIFY_SSL)

        if resp.status_code == 401:
            print("Token expired, refreshing...")
            token = get_token()
            headers["Authorization"] = f"Bearer {token}"
            resp = requests.put(url, json=value, headers=headers, verify=VERIFY_SSL)

        if not resp.ok:
            print(f"  [WARN] Row {i}: HTTP {resp.status_code} - {resp.text[:80]}")

        if (i + 1) % 500 == 0:
            print(f"  Uploaded {i + 1}/{total} rows...")
            time.sleep(0.5)  # avoid overwhelming the server

    print(f"Done. Uploaded {total} datapoints to '{ATTRIBUTE}'.")


def main():
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH, parse_dates=["time"])
    df = df[df["time"] >= START_DATE].sort_values("time").reset_index(drop=True)
    print(f"Rows to upload: {len(df)} ({df['time'].min()} to {df['time'].max()})")

    print("Authenticating...")
    token = get_token()
    print("Authenticated. Starting upload...")

    upload(token, df)


if __name__ == "__main__":
    main()
