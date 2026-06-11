"""Markdown summaries for single runs and batch comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def write_run_summary(
    output_path: Path,
    experiment_name: str,
    preprocess_name: str,
    model_name: str,
    split_metrics: dict[str, dict[str, float]],
) -> None:
    """Write a concise per-run markdown summary."""
    lines = [
        f"# {experiment_name} + {preprocess_name}",
        "",
        f"- Model: `{model_name}`",
        f"- Preprocessing: `{preprocess_name}`",
        "",
        "## Metrics",
        "",
        "| Split | MAE | RMSE | MAPE | sMAPE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split_name in ("train", "val", "test"):
        metrics = split_metrics[split_name]
        lines.append(
            f"| {split_name} | {metrics['MAE']:.4f} | {metrics['RMSE']:.4f} | "
            f"{metrics['MAPE']:.4f} | {metrics['sMAPE']:.4f} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_batch_summary(
    output_path: Path,
    results: Iterable[dict[str, object]],
    title: str = "Experiment Batch Summary",
    description: str = "This summary covers the assigned experiment runs.",
) -> None:
    """Write a markdown batch summary for a group of runs."""
    frame = pd.DataFrame(results)
    lines = [
        f"# {title}",
        "",
        description,
        "",
        "| Experiment | Preprocess | Test MAE | Test RMSE | Test MAPE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in frame.sort_values(["experiment_name", "preprocess"]).iterrows():
        lines.append(
            f"| {row['experiment_name']} | {row['preprocess']} | "
            f"{row['test_MAE']:.4f} | {row['test_RMSE']:.4f} | {row['test_MAPE']:.4f} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
