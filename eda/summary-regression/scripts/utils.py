from __future__ import annotations

import json
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import seaborn as sns
from matplotlib.figure import Figure

from eda.config import EDA_DIR, FIG_INTERACTIVE_DIR, FIG_STATIC_DIR, ROOT, REGRESSION_DIR, TABLE_DIR


def ensure_directories() -> None:
    for path in [ROOT / ".mplconfig", EDA_DIR, FIG_STATIC_DIR, FIG_INTERACTIVE_DIR, TABLE_DIR, REGRESSION_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update(
        {
            "figure.figsize": (12, 6),
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "savefig.bbox": "tight",
            "savefig.dpi": 180,
        }
    )


def save_figure(fig: Figure, filename: str) -> str:
    path = FIG_STATIC_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(EDA_DIR)).replace("\\", "/")


def save_plotly_figure(fig, filename: str, chart_meta: dict | None = None) -> str:
    path = FIG_INTERACTIVE_DIR / filename
    if chart_meta:
        subtitle = chart_meta.get("subtitle")
        if subtitle:
            fig.update_layout(
                title={
                    "text": f"{escape(chart_meta.get('title', 'Interactive Chart'))}<br><sup>{escape(subtitle)}</sup>",
                    "x": 0.02,
                    "xanchor": "left",
                }
            )
        fig.update_layout(
            margin={"t": 110, "r": 30, "b": 60, "l": 60},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        )
        chart_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
        info_sections = []
        sections = [
            ("What This Chart Is", chart_meta.get("what_it_is")),
            ("What Data Is Used", chart_meta.get("data_used")),
            ("How To Read It", chart_meta.get("how_to_read")),
            ("What It Shows", chart_meta.get("what_it_shows")),
            ("Why It Matters", chart_meta.get("why_it_matters")),
        ]
        for heading, text in sections:
            if text:
                info_sections.append(
                    f"<section><h2>{escape(heading)}</h2><p>{escape(text)}</p></section>"
                )
        if chart_meta.get("notes"):
            info_sections.append(
                "<section><h2>Important Notes</h2><ul>"
                + "".join(f"<li>{escape(note)}</li>" for note in chart_meta["notes"])
                + "</ul></section>"
            )
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(chart_meta.get('title', 'Interactive Chart'))}</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 24px auto; padding: 0 20px 40px; line-height: 1.55; color: #1f2933; }}
    h1 {{ margin-bottom: 0.3rem; }}
    .lede {{ color: #52606d; margin-top: 0; }}
    .chart-wrap {{ border: 1px solid #d9e2ec; border-radius: 12px; padding: 16px; background: #fff; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-top: 20px; }}
    section {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px 16px; background: #f8fafc; }}
    h2 {{ margin-top: 0; font-size: 1rem; }}
    p {{ margin: 0; }}
    ul {{ margin: 0; padding-left: 18px; }}
    code {{ background: #eef2f7; padding: 0.1rem 0.35rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{escape(chart_meta.get('title', 'Interactive Chart'))}</h1>
  <p class="lede">{escape(chart_meta.get('description', 'Interactive EDA chart generated from the project dataset.'))}</p>
  <div class="chart-wrap">{chart_html}</div>
  <div class="grid">
    {''.join(info_sections)}
  </div>
</body>
</html>
"""
        path.write_text(html, encoding="utf-8")
    else:
        fig.write_html(path, include_plotlyjs="cdn")
    return str(path.relative_to(EDA_DIR)).replace("\\", "/")


def write_interactive_index(charts: list[dict[str, str]]) -> str:
    items = []
    for chart in charts:
        items.append(
            f"<li><a href=\"{chart['filename']}\">{chart['title']}</a><br><small>{chart['description']}</small></li>"
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EDA Interactive Charts</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; line-height: 1.5; }}
    h1 {{ margin-bottom: 0.25rem; }}
    p {{ color: #333; }}
    li {{ margin-bottom: 1rem; }}
    a {{ font-weight: 600; }}
  </style>
</head>
<body>
  <h1>EDA Interactive Charts</h1>
  <p>Interactive Plotly outputs generated from <code>eda/eda.py</code>. Each page contains its own explanation of what the chart is, which dataset columns it uses, how to read it, and why it matters.</p>
  <ul>
    {''.join(items)}
  </ul>
</body>
</html>
"""
    path = FIG_INTERACTIVE_DIR / "index.html"
    path.write_text(html, encoding="utf-8")
    return str(path.relative_to(EDA_DIR)).replace("\\", "/")


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta,)):
        return str(value)
    if isinstance(value, (Path,)):
        return str(value)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join([header, divider] + rows)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    angle = 2 * np.pi * (df["hour"] / 24.0)
    df["hour_sin"] = np.sin(angle)
    df["hour_cos"] = np.cos(angle)
    return df


def infer_frequency_description(df: pd.DataFrame) -> dict:
    if df.empty or "time" not in df.columns:
        return {"description": "unknown", "mode": None, "min": None, "unique_steps": 0}
    sorted_times = df["time"].sort_values().drop_duplicates()
    deltas = sorted_times.diff().dropna()
    if deltas.empty:
        return {"description": "single timestamp", "mode": None, "min": None, "unique_steps": 0}
    mode = deltas.mode().iloc[0]
    min_delta = deltas.min()
    unique_steps = deltas.nunique()
    if unique_steps == 1:
        return {
            "description": f"regular ({mode})",
            "mode": str(mode),
            "min": str(min_delta),
            "unique_steps": int(unique_steps),
        }
    return {
        "description": f"mixed (mode {mode}, min {min_delta}, {unique_steps} unique steps)",
        "mode": str(mode),
        "min": str(min_delta),
        "unique_steps": int(unique_steps),
    }


def contiguous_gap_ranges(df: pd.DataFrame, expected_delta: pd.Timedelta) -> list[dict]:
    sorted_times = df["time"].sort_values().drop_duplicates().reset_index(drop=True)
    gap_rows = []
    for prev, curr in zip(sorted_times[:-1], sorted_times[1:]):
        diff = curr - prev
        if diff > expected_delta:
            missing_steps = int(diff / expected_delta) - 1
            gap_rows.append(
                {
                    "gap_after": prev,
                    "gap_before": curr,
                    "gap_duration": diff,
                    "missing_steps": missing_steps,
                }
            )
    return gap_rows


def summarize_dataset(name: str, path: Path, df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols + ["time"]]
    duplicate_rows = int(df.duplicated().sum())
    duplicate_timestamps = int(df["time"].duplicated().sum()) if "time" in df.columns else None
    missing_counts = df.isna().sum()
    return {
        "name": name,
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "date_start": df["time"].min() if "time" in df.columns else None,
        "date_end": df["time"].max() if "time" in df.columns else None,
        "frequency": infer_frequency_description(df),
        "duplicate_rows": duplicate_rows,
        "duplicate_timestamps": duplicate_timestamps,
        "missing_counts": missing_counts[missing_counts > 0].to_dict(),
    }


def select_complete_periods(df: pd.DataFrame, freq: str, year_cap: int | None = None) -> dict:
    work = df.copy()
    work["date"] = work.index
    if year_cap is not None:
        work = work[work.index.year <= year_cap]
    if freq == "H":
        month_counts = work.resample("MS").size()
        week_counts = work.resample("W-MON").size()
        day_counts = work.resample("D").size()
        month_start = month_counts[month_counts >= 24 * 28].index[0]
        week_start = week_counts[week_counts >= 24 * 7].index[0]
        day_start = day_counts[day_counts >= 24].index[0]
        return {
            "month_start": month_start,
            "month_end": month_start + pd.offsets.MonthEnd(0),
            "week_start": week_start,
            "week_end": week_start + pd.Timedelta(days=7),
            "day_start": day_start,
            "day_end": day_start + pd.Timedelta(days=1),
        }
    if freq == "15min":
        month_counts = work.resample("MS").size()
        week_counts = work.resample("W-MON").size()
        day_counts = work.resample("D").size()
        month_start = month_counts[month_counts >= 96 * 28].index[0]
        week_start = week_counts[week_counts >= 96 * 7].index[0]
        day_start = day_counts[day_counts >= 96].index[0]
        return {
            "month_start": month_start,
            "month_end": month_start + pd.offsets.MonthEnd(0),
            "week_start": week_start,
            "week_end": week_start + pd.Timedelta(days=7),
            "day_start": day_start,
            "day_end": day_start + pd.Timedelta(days=1),
        }
    raise ValueError(f"Unsupported frequency: {freq}")
