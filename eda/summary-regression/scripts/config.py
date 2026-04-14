from __future__ import annotations

import os
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

EDA_DIR = ROOT / "eda"
FIG_STATIC_DIR = EDA_DIR / "figures" / "static"
FIG_INTERACTIVE_DIR = EDA_DIR / "figures" / "interactive"
TABLE_DIR = EDA_DIR / "tables"
REGRESSION_DIR = EDA_DIR / "regression_outputs"

SOURCE_FILES = {
    "price_source": ROOT / "old" / "price_2015_2025.csv",
    "load_source": ROOT / "old" / "load_2015_2025.csv",
    "generation_source": ROOT / "old" / "generation_2015_2025.csv",
    "weather_source": ROOT / "old" / "weather_2015_2025_nl_avg_fixed.csv",
}

REFERENCE_FILES = {
    "final_dataset_full_raw": ROOT / "datasets" / "final_dataset_full_raw.csv",
    "final_dataset_full_clean": ROOT / "datasets" / "final_dataset_full_clean.csv",
    "legacy_weather_variant": ROOT / "old" / "weather_2015_2025.csv",
    "legacy_combined_weather_price": ROOT / "old" / "final_dataset.csv",
    "legacy_combined_with_temporal": ROOT / "old" / "final_dataset_with_features.csv",
}

TARGET = "price"
WEATHER_COLUMNS = [
    "temperature_2m",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
]
PRODUCTION_COLUMNS = ["total_load", "generation_forecast"]
EXOGENOUS_COLUMNS = WEATHER_COLUMNS + PRODUCTION_COLUMNS
CALENDAR_COLUMNS = ["hour_sin", "hour_cos", "day_of_week", "month", "is_weekend"]

NEGATIVE_PRICE_FLOOR = -500.0
PRICE_UPPER_SANITY = 1000.0
WEATHER_RULES = {
    "temperature_2m": (-40.0, 50.0),
    "cloud_cover": (0.0, 1.0),
    "wind_speed_10m": (0.0, 60.0),
    "shortwave_radiation": (0.0, 1400.0),
}
SYSTEM_RULES = {
    "total_load": (0.0, None),
    "generation_forecast": (0.0, None),
}
