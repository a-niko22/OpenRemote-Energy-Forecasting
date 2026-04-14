from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from eda.config import PRODUCTION_COLUMNS, TARGET, WEATHER_COLUMNS
from eda.utils import save_figure


def apply_time_axis(ax: Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def plot_line(
    df: pd.DataFrame,
    y: str,
    title: str,
    ylabel: str,
    filename: str,
    xlim: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> str:
    fig, ax = plt.subplots(figsize=(14, 5.5))
    work = df if xlim is None else df.loc[xlim[0] : xlim[1]]
    ax.plot(work.index, work[y], linewidth=1.1, color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    apply_time_axis(ax)
    return save_figure(fig, filename)


def plot_box(df: pd.DataFrame, by_col: str, y: str, title: str, xlabel: str, ylabel: str, filename: str) -> str:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x=by_col, y=y, ax=ax, color="#9ecae1", fliersize=1.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return save_figure(fig, filename)


def plot_histogram(series: pd.Series, title: str, xlabel: str, filename: str, bins: int = 50) -> str:
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.histplot(series.dropna(), bins=bins, kde=True, ax=ax, color="#4c72b0")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    return save_figure(fig, filename)


def plot_missingness_bar(df: pd.DataFrame, filename: str) -> str:
    plot_df = df.isna().mean().sort_values(ascending=False).mul(100).rename("missing_pct")
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", ax=ax, color="#c44e52")
    ax.set_title("Missingness by Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing values (%)")
    ax.tick_params(axis="x", rotation=30)
    return save_figure(fig, filename)


def plot_missingness_heatmap(df: pd.DataFrame, filename: str) -> str:
    sample = df.copy()
    if len(sample) > 3000:
        sample = sample.iloc[:3000]
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(sample.isna().T, cmap="viridis", cbar=False, ax=ax)
    ax.set_title("Missingness Heatmap (first 3,000 rows)")
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Column")
    return save_figure(fig, filename)


def plot_source_coverage(source_summaries: list[dict], filename: str) -> str:
    plot_df = pd.DataFrame(
        {
            "dataset": [row["name"] for row in source_summaries],
            "date_start": [row["date_start"] for row in source_summaries],
            "date_end": [row["date_end"] for row in source_summaries],
        }
    ).sort_values("date_start")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for idx, row in plot_df.reset_index(drop=True).iterrows():
        ax.hlines(y=idx, xmin=row["date_start"], xmax=row["date_end"], linewidth=8, color="#4c72b0")
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["dataset"])
    ax.set_title("Source Coverage Timeline")
    ax.set_xlabel("Time")
    apply_time_axis(ax)
    return save_figure(fig, filename)


def plot_gap_events(gap_table: pd.DataFrame, filename: str) -> str:
    plot_df = gap_table.copy()
    plot_df["gap_hours"] = plot_df["gap_duration"] / pd.Timedelta(hours=1)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.scatterplot(data=plot_df, x="gap_after", y="source", size="gap_hours", hue="gap_hours", palette="flare", ax=ax)
    ax.set_title("Detected Gap Events by Source")
    ax.set_xlabel("Gap start time")
    ax.set_ylabel("Source")
    apply_time_axis(ax)
    return save_figure(fig, filename)


def plot_price_frequency_transition(price_df: pd.DataFrame, filename: str) -> str:
    daily_counts = price_df.resample("D").size().rename("observations_per_day")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(daily_counts.index, daily_counts.values, color="#c44e52", linewidth=1.2)
    ax.axvline(pd.Timestamp("2025-01-01"), color="black", linestyle="--", linewidth=1.0, label="2025 boundary")
    ax.set_title("Price Frequency Transition: Daily Row Counts")
    ax.set_xlabel("Time")
    ax.set_ylabel("Rows per day")
    ax.legend()
    apply_time_axis(ax)
    return save_figure(fig, filename)


def plot_weekday_weekend(df: pd.DataFrame, filename: str) -> str:
    grouped = df.groupby([df["is_weekend"].map({0: "Weekday", 1: "Weekend"}), "hour"])[TARGET].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=grouped, x="hour", y=TARGET, hue="is_weekend", ax=ax, linewidth=2.0)
    ax.set_title("Average Hourly Price: Weekday vs Weekend")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average price (EUR/MWh)")
    ax.legend(title="Day type")
    return save_figure(fig, filename)


def plot_acf_pacf(series: pd.Series, nlags: int, acf_filename: str, pacf_filename: str) -> tuple[str, str]:
    clean = series.dropna()
    fig_acf, ax_acf = plt.subplots(figsize=(12, 4.5))
    plot_acf(clean, lags=nlags, ax=ax_acf)
    ax_acf.set_title("Autocorrelation of Hourly Electricity Price")
    acf_path = save_figure(fig_acf, acf_filename)

    fig_pacf, ax_pacf = plt.subplots(figsize=(12, 4.5))
    plot_pacf(clean, lags=min(nlags, 120), ax=ax_pacf, method="ywm")
    ax_pacf.set_title("Partial Autocorrelation of Hourly Electricity Price")
    pacf_path = save_figure(fig_pacf, pacf_filename)
    return acf_path, pacf_path


def plot_seasonal_decomposition(series: pd.Series, filename: str, period: int = 24) -> str:
    clean = series.dropna()
    sample = clean.iloc[: min(len(clean), period * 90)]
    decomposition = seasonal_decompose(sample, model="additive", period=period)
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    return save_figure(fig, filename)


def plot_weather_timeseries(df: pd.DataFrame, filename: str) -> str:
    daily = df[WEATHER_COLUMNS].resample("D").mean()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for ax, col in zip(axes.ravel(), WEATHER_COLUMNS):
        ax.plot(daily.index, daily[col], color="#4c72b0")
        ax.set_title(col)
        ax.set_xlabel("Time")
        ax.set_ylabel(col)
        apply_time_axis(ax)
    fig.suptitle("Weather Features Over Time (Daily Mean)", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, filename)


def plot_production_timeseries(df: pd.DataFrame, filename: str) -> str:
    daily = df[PRODUCTION_COLUMNS].resample("D").mean()
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for ax, col in zip(axes.ravel(), PRODUCTION_COLUMNS):
        ax.plot(daily.index, daily[col], color="#55a868")
        ax.set_title(col)
        ax.set_xlabel("Time")
        ax.set_ylabel(col)
        apply_time_axis(ax)
    fig.suptitle("Production / System Features Over Time (Daily Mean)", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, filename)


def plot_feature_histograms(df: pd.DataFrame, columns: list[str], title: str, filename: str) -> str:
    cols = len(columns)
    fig, axes = plt.subplots(cols, 1, figsize=(11, 3.2 * cols))
    if cols == 1:
        axes = [axes]
    for ax, col in zip(axes, columns):
        sns.histplot(df[col].dropna(), bins=50, kde=True, ax=ax, color="#4c72b0")
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, filename)


def plot_correlation_heatmap(df: pd.DataFrame, filename: str) -> str:
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap for Price, Weather, Production, and Calendar Features")
    return save_figure(fig, filename)


def plot_scatter_grid(df: pd.DataFrame, target_col: str, feature_cols: list[str], filename: str) -> str:
    ncols = 2
    nrows = int(np.ceil(len(feature_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)
    sample = df[[target_col] + feature_cols].dropna()
    if len(sample) > 15000:
        sample = sample.sample(15000, random_state=42)
    for ax, feature in zip(axes, feature_cols):
        sns.scatterplot(data=sample, x=feature, y=target_col, alpha=0.15, s=15, ax=ax, color="#4c72b0")
        ax.set_title(f"{feature} vs {target_col}")
        ax.set_xlabel(feature)
        ax.set_ylabel(target_col)
    for ax in axes[len(feature_cols) :]:
        ax.axis("off")
    fig.suptitle("Price vs Exogenous Feature Scatter Plots", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(fig, filename)


def plot_lagged_correlations(lagged_corr: pd.DataFrame, filename: str) -> str:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=lagged_corr, x="lag_hours", y="correlation", hue="feature", ax=ax)
    ax.set_title("Lagged Correlation Screen: Exogenous Variables vs Price")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Correlation with price")
    ax.legend(title="Feature")
    return save_figure(fig, filename)


def plot_missingness_over_time(df: pd.DataFrame, filename: str) -> str:
    daily_missing = df.isna().sum(axis=1).resample("D").sum()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(daily_missing.index, daily_missing.values, color="#c44e52")
    ax.set_title("Missing Exogenous Values Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Missing values per day")
    apply_time_axis(ax)
    return save_figure(fig, filename)


def plot_anomalies(df: pd.DataFrame, column: str, filename: str) -> str:
    series = df[column].dropna()
    mad = np.median(np.abs(series - series.median()))
    scale = mad * 1.4826 if mad else 1
    z = (series - series.median()) / scale
    flagged = z.abs() > 4
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(series.index, series.values, color="#4c72b0", linewidth=0.8, label=column)
    ax.scatter(series.index[flagged], series[flagged], color="#c44e52", s=12, label="flagged anomalies")
    ax.set_title(f"Anomaly Candidates: {column}")
    ax.set_xlabel("Time")
    ax.set_ylabel(column)
    ax.legend()
    apply_time_axis(ax)
    return save_figure(fig, filename)


def plot_outlier_boxplots(df: pd.DataFrame, columns: list[str], filename: str) -> str:
    plot_df = df[columns].copy().melt(var_name="feature", value_name="value").dropna()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=plot_df, x="feature", y="value", ax=ax, color="#9ecae1", fliersize=0.8)
    ax.set_title("Outlier Comparison Across Price and Exogenous Variables")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=25)
    return save_figure(fig, filename)
