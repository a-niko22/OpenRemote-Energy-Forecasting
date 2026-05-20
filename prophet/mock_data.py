"""Generate mock hourly tariff data for development when API is unavailable."""
import numpy as np
import pandas as pd

def generate_mock_tariff_data(days: int = 365, freq: str = "h") -> pd.DataFrame:
    np.random.seed(42)
    periods = days * 24
    timestamps = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=periods, freq=freq)

    # Daily pattern: cheap at night, expensive morning/evening peaks
    hour = timestamps.hour
    daily = (
        -2 * np.cos(2 * np.pi * hour / 24)          # base daily cycle
        + 1.5 * np.cos(2 * np.pi * (hour - 8) / 12) # morning/evening peaks
    )

    # Weekly pattern: lower on weekends
    weekly = -0.5 * (timestamps.dayofweek >= 5).astype(float)

    # Seasonal trend
    day_of_year = timestamps.dayofyear
    seasonal = np.sin(2 * np.pi * day_of_year / 365)

    # Noise
    noise = np.random.normal(0, 0.3, periods)

    price = 0.15 + 0.05 * (daily + weekly + seasonal + noise)
    price = np.clip(price, 0.01, 0.50)  # realistic EUR/kWh bounds

    return pd.DataFrame({"ds": timestamps, "y": price})


if __name__ == "__main__":
    df = generate_mock_tariff_data()
    df.to_csv("prophet/mock_tariff.csv", index=False)
    print(f"Saved {len(df)} rows to prophet/mock_tariff.csv")
    print(df.tail())
