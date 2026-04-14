from __future__ import annotations

import numpy as np
import pandas as pd

from eda.config import (
    EXOGENOUS_COLUMNS,
    NEGATIVE_PRICE_FLOOR,
    PRICE_UPPER_SANITY,
    SYSTEM_RULES,
    TARGET,
    WEATHER_RULES,
)


def compute_lagged_correlations(df: pd.DataFrame, target_col: str, feature_cols: list[str], max_lag: int = 48) -> pd.DataFrame:
    rows = []
    for feature in feature_cols:
        for lag in range(max_lag + 1):
            shifted = df[feature].shift(lag)
            corr = df[target_col].corr(shifted)
            rows.append({"feature": feature, "lag_hours": lag, "correlation": corr})
    return pd.DataFrame(rows)


def iqr_outlier_mask(series: pd.Series) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return pd.Series(False, index=series.index)
    q1, q3 = clean.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=series.index)
    return (series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))


def robust_anomaly_mask(series: pd.Series, threshold: float = 4.0) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return pd.Series(False, index=series.index)
    med = clean.median()
    mad = np.median(np.abs(clean - med))
    if mad == 0:
        return pd.Series(False, index=series.index)
    robust_z = 0.6745 * (series - med) / mad
    return robust_z.abs() > threshold


def build_dataset_inventory(source_summaries: list[dict], reference_summaries: list[dict], price_break_ts: pd.Timestamp | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for summary in source_summaries + reference_summaries:
        rows.append(
            {
                "dataset": summary["name"],
                "rows": summary["rows"],
                "columns": len(summary["columns"]),
                "date_start": summary["date_start"],
                "date_end": summary["date_end"],
                "frequency": summary["frequency"]["description"],
                "duplicate_rows": summary["duplicate_rows"],
                "duplicate_timestamps": summary["duplicate_timestamps"],
                "missing_columns": ", ".join(summary["missing_counts"].keys()) or "None",
            }
        )
    notes = [
        {"item": "Target column", "detail": "price"},
        {"item": "Primary timestamp column", "detail": "time"},
        {"item": "Primary exogenous variables", "detail": ", ".join(EXOGENOUS_COLUMNS)},
        {"item": "Selected weather source", "detail": "old/weather_2015_2025_nl_avg_fixed.csv"},
        {
            "item": "Excluded weather variant",
            "detail": "old/weather_2015_2025.csv uses a different timestamp format and different scaling such as cloud_cover 34 instead of 0.236.",
        },
        {
            "item": "Price frequency break",
            "detail": f"First quarter-hour timestamp appears at {price_break_ts}" if price_break_ts is not None else "No frequency break detected",
        },
    ]
    return pd.DataFrame(rows), pd.DataFrame(notes)


def data_quality_checks(hourly_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    checks = []
    for col in [TARGET] + EXOGENOUS_COLUMNS:
        series = hourly_panel[col]
        outliers = iqr_outlier_mask(series)
        anomalies = robust_anomaly_mask(series)
        checks.append(
            {
                "column": col,
                "missing_count": int(series.isna().sum()),
                "missing_pct": round(series.isna().mean() * 100, 3),
                "negative_count": int((series < 0).sum()) if pd.api.types.is_numeric_dtype(series) else None,
                "zero_count": int((series == 0).sum()) if pd.api.types.is_numeric_dtype(series) else None,
                "iqr_outlier_count": int(outliers.sum()),
                "robust_anomaly_count": int(anomalies.sum()),
                "std": round(float(series.std()), 4) if pd.api.types.is_numeric_dtype(series) else None,
                "unique_values": int(series.nunique(dropna=True)),
            }
        )

    suspicious_rows = []
    for col, bounds in WEATHER_RULES.items():
        lower, upper = bounds
        series = hourly_panel[col]
        mask = series < lower
        if upper is not None:
            mask |= series > upper
        suspicious_rows.append({"column": col, "rule": f"[{lower}, {upper}]", "suspicious_count": int(mask.sum())})
    for col, bounds in SYSTEM_RULES.items():
        lower, upper = bounds
        series = hourly_panel[col]
        mask = series < lower
        if upper is not None:
            mask |= series > upper
        suspicious_rows.append({"column": col, "rule": f"[{lower}, {upper}]", "suspicious_count": int(mask.sum())})
    price_mask = (hourly_panel[TARGET] < NEGATIVE_PRICE_FLOOR) | (hourly_panel[TARGET] > PRICE_UPPER_SANITY)
    suspicious_rows.append(
        {
            "column": TARGET,
            "rule": f"[{NEGATIVE_PRICE_FLOOR}, {PRICE_UPPER_SANITY}]",
            "suspicious_count": int(price_mask.sum()),
        }
    )
    return pd.DataFrame(checks), pd.DataFrame(suspicious_rows)


def feature_signal_screen(hourly_panel: pd.DataFrame, lagged_corr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in EXOGENOUS_COLUMNS:
        series = hourly_panel[feature]
        best_row = lagged_corr[lagged_corr["feature"] == feature].iloc[lagged_corr[lagged_corr["feature"] == feature]["correlation"].abs().argmax()]
        outlier_mask = iqr_outlier_mask(series)
        rows.append(
            {
                "feature": feature,
                "missing_pct": round(series.isna().mean() * 100, 3),
                "std": round(float(series.std()), 4),
                "lag1_autocorr": round(float(series.autocorr(lag=1)), 4),
                "corr_with_price_t": round(float(hourly_panel[TARGET].corr(series)), 4),
                "best_lag_corr": round(float(best_row["correlation"]), 4),
                "best_abs_corr": round(float(abs(best_row["correlation"])), 4),
                "best_lag_hours": int(best_row["lag_hours"]),
                "iqr_outlier_pct": round(float(outlier_mask.mean() * 100), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("best_abs_corr", ascending=False)


def build_leakage_table() -> pd.DataFrame:
    rows = [
        {
            "feature": "calendar features (hour, day_of_week, month, is_weekend, cyclical hour)",
            "category": "safe",
            "reason": "Calendar information is known at forecast time.",
        },
        {
            "feature": "lagged prices created only from historical values",
            "category": "safe",
            "reason": "Past prices are available at forecast time.",
        },
        {
            "feature": "rolling features computed only from past windows",
            "category": "safe",
            "reason": "Past-only windows do not leak future target information.",
        },
        {
            "feature": "temperature_2m, cloud_cover, wind_speed_10m, shortwave_radiation",
            "category": "safe only if forecasted",
            "reason": "Observed future weather is not available for a 48-hour-ahead forecast unless forecast data is used.",
        },
        {
            "feature": "generation_forecast",
            "category": "unclear",
            "reason": "The name suggests a forecast, but the dataset does not include forecast issuance timestamps or vintages.",
        },
        {
            "feature": "total_load",
            "category": "safe only if forecasted",
            "reason": "Observed future load would leak; forecasted load could be safe if only the ex-ante forecast is used.",
        },
        {
            "feature": "future observed production or weather values aligned directly with the target horizon",
            "category": "unsafe / leaky",
            "reason": "These values would not be known when issuing the forecast.",
        },
        {
            "feature": "centered rolling windows or features derived from future prices",
            "category": "unsafe / leaky",
            "reason": "They use future observations from the prediction period.",
        },
    ]
    return pd.DataFrame(rows)


def summarize_frequency_transition(price_df: pd.DataFrame) -> dict:
    non_hourly = price_df[price_df["time"].dt.minute != 0]
    break_ts = non_hourly["time"].min() if not non_hourly.empty else None
    hourly_only_end = price_df.loc[price_df["time"] < break_ts, "time"].max() if break_ts is not None else price_df["time"].max()
    top_of_hour = price_df[price_df["time"].dt.minute == 0]["time"].max()
    quarter_hour_counts = (
        price_df.assign(is_quarter=price_df["time"].dt.minute.isin([15, 30, 45]))
        .groupby(price_df["time"].dt.year)["is_quarter"]
        .sum()
        .to_dict()
    )
    return {
        "hourly_only_segment_end": hourly_only_end,
        "last_top_of_hour_timestamp": top_of_hour,
        "first_non_hourly_timestamp": break_ts,
        "quarter_hour_counts_by_year": {str(k): int(v) for k, v in quarter_hour_counts.items()},
    }


def audit_joined_datasets(hourly_panel: pd.DataFrame, raw_join: pd.DataFrame, clean_join: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned_set = set(hourly_panel["time"])
    raw_set = set(raw_join["time"])
    clean_set = set(clean_join["time"])
    rows = [
        {"dataset": "hourly_panel_raw", "rows": len(hourly_panel), "missing_times_vs_hourly_panel": 0, "extra_times_vs_hourly_panel": 0},
        {
            "dataset": "final_dataset_full_raw",
            "rows": len(raw_join),
            "missing_times_vs_hourly_panel": len(aligned_set - raw_set),
            "extra_times_vs_hourly_panel": len(raw_set - aligned_set),
        },
        {
            "dataset": "final_dataset_full_clean",
            "rows": len(clean_join),
            "missing_times_vs_hourly_panel": len(aligned_set - clean_set),
            "extra_times_vs_hourly_panel": len(clean_set - aligned_set),
        },
    ]
    removed_times = sorted(list(raw_set - clean_set))
    return pd.DataFrame(rows), pd.DataFrame({"time_removed_in_clean": removed_times})


def choose_promising_features(signal_screen: pd.DataFrame, regression_metrics: pd.DataFrame) -> list[str]:
    candidates = signal_screen.dropna(subset=["best_abs_corr"]).copy()
    candidates = candidates[candidates["missing_pct"] < 5]
    top_corr = candidates.head(4)["feature"].tolist()
    simple_models = regression_metrics[regression_metrics["model_type"] == "simple"].sort_values("adj_r2", ascending=False)
    top_simple = simple_models.head(4)["primary_feature"].tolist()
    ordered = []
    for feature in top_corr + top_simple:
        if feature and feature not in ordered:
            ordered.append(feature)
    return ordered[:5]


def recommendation_paragraphs(promising_features: list[str], signal_screen: pd.DataFrame, quality_table: pd.DataFrame) -> dict:
    weak_features = signal_screen.sort_values("best_abs_corr", ascending=True).head(2)["feature"].tolist()
    most_missing = quality_table.sort_values("missing_pct", ascending=False).head(2)["column"].tolist()
    return {
        "promising": ", ".join(promising_features) if promising_features else "None identified robustly",
        "weak": ", ".join(weak_features),
        "missing": ", ".join(most_missing),
    }
