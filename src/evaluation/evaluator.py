"""Model evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.data.windowing import ScalerBundle, WindowSplit
from src.training.metrics import regression_metrics


@dataclass
class SplitEvaluation:
    """Predictions, targets, and metrics for one split."""

    predictions: np.ndarray
    targets: np.ndarray
    metrics: dict[str, float]


@dataclass
class EvaluationBundle:
    """Evaluation results across train, val, and test."""

    splits: Dict[str, SplitEvaluation]


def inverse_transform_targets(values: np.ndarray, scaler_bundle: ScalerBundle) -> np.ndarray:
    """Inverse transform a batch of scaled direct forecasts."""
    if values.size == 0:
        return values
    shape = values.shape
    restored = scaler_bundle.target_scaler.inverse_transform(values.reshape(-1, 1))
    return restored.reshape(shape)


def evaluate_predictions(
    scaled_predictions: Dict[str, np.ndarray],
    scaled_targets: Dict[str, np.ndarray],
    scaler_bundle: ScalerBundle,
) -> EvaluationBundle:
    """Evaluate all splits in original price units."""
    results: Dict[str, SplitEvaluation] = {}
    for split_name, predictions in scaled_predictions.items():
        targets = scaled_targets[split_name]
        restored_predictions = inverse_transform_targets(predictions, scaler_bundle)
        restored_targets = inverse_transform_targets(targets, scaler_bundle)
        results[split_name] = SplitEvaluation(
            predictions=restored_predictions,
            targets=restored_targets,
            metrics=regression_metrics(restored_predictions, restored_targets),
        )
    return EvaluationBundle(splits=results)


def build_prediction_rows(
    split_name: str,
    split_data: WindowSplit,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> list[dict[str, object]]:
    """Flatten predictions into a long-form table."""
    rows: list[dict[str, object]] = []
    for sample_index in range(predictions.shape[0]):
        for horizon_step in range(predictions.shape[1]):
            rows.append(
                {
                    "split": split_name,
                    "sample_id": sample_index,
                    "forecast_step": horizon_step + 1,
                    "target_timestamp": str(split_data.target_timestamps[sample_index, horizon_step]),
                    "actual": float(targets[sample_index, horizon_step]),
                    "predicted": float(predictions[sample_index, horizon_step]),
                    "residual": float(predictions[sample_index, horizon_step] - targets[sample_index, horizon_step]),
                }
            )
    return rows
