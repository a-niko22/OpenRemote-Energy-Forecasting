import os
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from dotenv import load_dotenv

load_dotenv()

FORECAST_HOURS = 48  # how far ahead to predict


def load_data_from_api() -> pd.DataFrame:
    """Load tariff data from OpenRemote API and return Prophet-formatted DataFrame."""
    import json
    import requests

    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    host_name = os.getenv("HOST_NAME")
    realm_name = os.getenv("REALM_NAME")
    asset_id = os.getenv("ASSET_ID")
    attribute_name = os.getenv("ATTRIBUTE_NAME")

    token_url = f"https://{host_name}/auth/realms/{realm_name}/protocol/openid-connect/token"
    token_resp = requests.post(token_url, data={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    })
    token_resp.raise_for_status()
    access_token = token_resp.json()["access_token"]

    from datetime import datetime, timedelta
    end = datetime.now().date() + timedelta(days=2)
    start = end - timedelta(days=365)

    api_url = f"https://{host_name}/api/{realm_name}/asset/datapoint/{asset_id}/{attribute_name}"
    resp = requests.post(api_url, headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }, data=json.dumps({
        "fromTimestamp": 0,
        "toTimestamp": 0,
        "fromTime": f"{start}T00:00:00.000Z",
        "toTime": f"{end}T00:00:00.000Z",
        "type": "string",
    }))
    if not resp.ok:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text[:200]}")

    points = resp.json()
    df = pd.DataFrame({
        "ds": pd.to_datetime([p["x"] for p in points], unit="ms"),
        "y": [float(p["y"]) for p in points],
    })
    return df


def load_data_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ds"])
    return df[["ds", "y"]]


def build_model() -> Prophet:
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="multiplicative",  # tariffs scale multiplicatively
        changepoint_prior_scale=0.1,        # regularize trend changes
    )
    return model


def train_and_forecast(df: pd.DataFrame) -> tuple[Prophet, pd.DataFrame]:
    model = build_model()
    model.fit(df)

    future = model.make_future_dataframe(periods=FORECAST_HOURS, freq="h")
    forecast = model.predict(future)
    return model, forecast


def evaluate(model: Prophet, df: pd.DataFrame) -> pd.DataFrame:
    cv_results = cross_validation(
        model,
        initial="180 days",
        period="30 days",
        horizon="48 hours",
    )
    return performance_metrics(cv_results)


def plot_results(model: Prophet, forecast: pd.DataFrame, df: pd.DataFrame):
    fig1 = model.plot(forecast)
    plt.title("Tariff Forecast")
    plt.tight_layout()
    plt.savefig("prophet/forecast.png", dpi=150)
    print("Saved prophet/forecast.png")

    fig2 = model.plot_components(forecast)
    plt.tight_layout()
    plt.savefig("prophet/forecast_components.png", dpi=150)
    print("Saved prophet/forecast_components.png")


if __name__ == "__main__":
    import sys

    csv_path = "prophet/mock_tariff.csv"
    use_mock = "--mock" in sys.argv or os.path.exists(csv_path) and "--api" not in sys.argv

    if use_mock:
        print(f"Loading mock data from {csv_path}")
        if not os.path.exists(csv_path):
            from mock_data import generate_mock_tariff_data
            df = generate_mock_tariff_data()
            df.to_csv(csv_path, index=False)
        df = load_data_from_csv(csv_path)
    else:
        print("Loading data from API...")
        df = load_data_from_api()

    print(f"Data: {len(df)} rows, {df['ds'].min()} → {df['ds'].max()}")

    model, forecast = train_and_forecast(df)
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(FORECAST_HOURS))

    plot_results(model, forecast, df)
