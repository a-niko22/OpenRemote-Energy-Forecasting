"""Plotting utilities for run-level and batch-level evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None
    from PIL import Image, ImageDraw


def _draw_simple_chart(
    series_collection,
    colors,
    labels,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    width, height = 1200, 500
    margin = 80
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    flat_values = [float(value) for series in series_collection for value in series]
    min_y = min(flat_values) if flat_values else 0.0
    max_y = max(flat_values) if flat_values else 1.0
    if max_y == min_y:
        max_y += 1.0

    draw.rectangle([margin, margin, width - margin, height - margin], outline="black", width=2)
    draw.text((margin, 20), title, fill="black")
    draw.text((width // 2, height - 40), x_label, fill="black")
    draw.text((10, height // 2), y_label, fill="black")

    for series, color, label in zip(series_collection, colors, labels):
        if len(series) < 2:
            continue
        points = []
        for index, value in enumerate(series):
            x = margin + (index / max(1, len(series) - 1)) * (width - 2 * margin)
            y = height - margin - ((float(value) - min_y) / (max_y - min_y)) * (height - 2 * margin)
            points.append((x, y))
        draw.line(points, fill=color, width=3)
        draw.text((width - margin - 180, margin + 20 * labels.index(label)), label, fill=color)

    image.save(output_path)


def plot_loss(history: dict[str, list[float]], output_path: Path) -> None:
    """Plot train and validation loss across epochs."""
    if plt is None:
        _draw_simple_chart(
            series_collection=[history["train_loss"], history["val_loss"]],
            colors=["blue", "orange"],
            labels=["train", "val"],
            title="Training history",
            x_label="Epoch",
            y_label="MSE loss",
            output_path=output_path,
        )
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_title("Training history")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_predictions(predictions, targets, output_path: Path, max_windows: int = 3) -> None:
    """Plot representative prediction windows."""
    if len(predictions) == 0:
        return
    if plt is None:
        _draw_simple_chart(
            series_collection=[targets[0], predictions[0]],
            colors=["black", "blue"],
            labels=["Actual", "Predicted"],
            title="Representative prediction window",
            x_label="Forecast step",
            y_label="Price",
            output_path=output_path,
        )
        return
    windows = min(max_windows, len(predictions))
    fig, axes = plt.subplots(windows, 1, figsize=(12, 3 * windows), sharex=False)
    if windows == 1:
        axes = [axes]
    for index, axis in enumerate(axes):
        axis.plot(targets[index], label="Actual", color="black", linewidth=1.8)
        axis.plot(predictions[index], label="Predicted", color="tab:blue", linewidth=1.4)
        axis.set_title(f"Test sample {index + 1}")
        axis.set_ylabel("Price")
        axis.legend()
    axes[-1].set_xlabel("Forecast step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_residuals(predictions, targets, output_path: Path) -> None:
    """Plot residuals over all test points."""
    if len(predictions) == 0:
        return
    residuals = (predictions - targets).reshape(-1)
    if plt is None:
        zero_line = [0.0 for _ in residuals]
        _draw_simple_chart(
            series_collection=[residuals, zero_line],
            colors=["red", "black"],
            labels=["Residual", "Zero"],
            title="Residuals",
            x_label="Flattened prediction index",
            y_label="Prediction error",
            output_path=output_path,
        )
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(range(len(residuals)), residuals, s=8, alpha=0.5, color="tab:red")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Residuals")
    ax.set_xlabel("Flattened prediction index")
    ax.set_ylabel("Prediction error")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_batch_comparison(summary_frame: pd.DataFrame, output_path: Path) -> None:
    """Plot a grouped comparison of test MAE and RMSE across runs."""
    if summary_frame.empty:
        return
    plot_frame = summary_frame.copy()
    plot_frame["label"] = plot_frame["experiment_name"] + " + " + plot_frame["preprocess"]
    if plt is None:
        _draw_simple_chart(
            series_collection=[plot_frame["test_MAE"].tolist(), plot_frame["test_RMSE"].tolist()],
            colors=["blue", "orange"],
            labels=["Test MAE", "Test RMSE"],
            title="Batch comparison",
            x_label="Run index",
            y_label="Metric value",
            output_path=output_path,
        )
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(plot_frame["label"], plot_frame["test_MAE"], color="tab:blue")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Test MAE by run")

    axes[1].bar(plot_frame["label"], plot_frame["test_RMSE"], color="tab:orange")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Test RMSE by run")
    axes[1].set_xlabel("Run")

    for axis in axes:
        axis.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
