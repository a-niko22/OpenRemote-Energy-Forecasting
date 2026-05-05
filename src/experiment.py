"""Experiment orchestration for Exp.1.a and Exp.1.b."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.data.dataset import load_time_series_data
from src.data.preprocessing import apply_preprocessing
from src.data.windowing import build_dataloaders, build_windowed_splits
from src.evaluation.evaluator import build_prediction_rows, evaluate_predictions
from src.evaluation.plots import (
    plot_batch_comparison,
    plot_loss,
    plot_predictions,
    plot_residuals,
)
from src.evaluation.reports import write_batch_summary, write_run_summary
from src.models import MODEL_REGISTRY
from src.training.trainer import Trainer
from src.utils.config import (
    ExperimentConfig,
    experiment_config_to_dict,
    load_experiment_config,
)
from src.utils.io import ensure_dir, save_dataframe, save_joblib, save_json, save_yaml
from src.utils.logging import build_logger
from src.utils.seed import set_global_seed


def resolve_device(device_name: str) -> str:
    """Resolve auto device selection."""
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def prepare_run_config(
    config_path: str | Path,
    preprocess_name: str,
    data_path: str | None = None,
    device: str | None = None,
    output_root: str | None = None,
    seed: int | None = None,
) -> ExperimentConfig:
    """Load the YAML config and apply CLI overrides."""
    config = load_experiment_config(config_path)

    if data_path is not None:
        config.data.path = data_path
    if device is not None:
        config.device = device
    if output_root is not None:
        config.outputs.root_dir = output_root
    if seed is not None:
        config.seed = seed

    if preprocess_name not in {"norm", "wavelet", "patch", "exp2_standard", "exp2_fft", "exp2_kernel"}:
        raise ValueError(f"Unsupported preprocessing strategy: {preprocess_name}")

    return config


def run_experiment(
    config_path: str | Path,
    preprocess_name: str,
    data_path: str | None = None,
    device: str | None = None,
    output_root: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run one experiment and persist all requested artifacts."""
    config = prepare_run_config(
        config_path=config_path,
        preprocess_name=preprocess_name,
        data_path=data_path,
        device=device,
        output_root=output_root,
        seed=seed,
    )
    set_global_seed(config.seed)

    run_root = ensure_dir(
        Path(config.outputs.root_dir) / config.experiment_name / preprocess_name
    )
    log_dir = ensure_dir(run_root / "logs")
    metrics_dir = ensure_dir(run_root / "metrics")
    plots_dir = ensure_dir(run_root / "plots")
    artifacts_dir = ensure_dir(run_root / "artifacts")

    logger = build_logger(
        f"{config.experiment_name}_{preprocess_name}", log_dir / "run.log"
    )
    logger.info(
        "Starting run for %s with preprocess=%s",
        config.experiment_name,
        preprocess_name,
    )
    logger.info("Resolved device: %s", resolve_device(config.device))

    loaded_data = load_time_series_data(config.data)

    if preprocess_name == "exp2_standard":
        from src.data.exp2_preprocessing import apply_exp2_standard_preprocessing

        preprocessed_frame, new_feature_cols = apply_exp2_standard_preprocessing(
            loaded_data.frame, config.data.target_col
        )
        loaded_data = replace(
            loaded_data, frame=preprocessed_frame, feature_cols=new_feature_cols
        )
    elif preprocess_name == "exp2_fft":
        from src.data.exp2_fft_preprocessing import apply_exp2_fft_preprocessing

        preprocessed_frame, new_feature_cols = apply_exp2_fft_preprocessing(
            loaded_data.frame, config.data.target_col
        )
        loaded_data = replace(
            loaded_data, frame=preprocessed_frame, feature_cols=new_feature_cols
        )
    elif preprocess_name == "exp2_kernel":
        from src.data.exp2_kernel_preprocessing import apply_exp2_kernel_preprocessing

        preprocessed_frame, new_feature_cols = apply_exp2_kernel_preprocessing(
            loaded_data.frame, config.data.target_col
        )
        loaded_data = replace(
            loaded_data, frame=preprocessed_frame, feature_cols=new_feature_cols
        )

    config.data.feature_cols = loaded_data.feature_cols
    logger.info(
        "Loaded dataset with %d rows, target=%s, features=%s",
        len(loaded_data.frame),
        loaded_data.target_col,
        loaded_data.feature_cols,
    )

    windowed_bundle = build_windowed_splits(loaded_data.frame, config)
    preprocessed_bundle = apply_preprocessing(
        splits=windowed_bundle.splits,
        input_feature_names=windowed_bundle.input_feature_names,
        preprocess_name=preprocess_name,
        config=config,
    )

    dataloaders = build_dataloaders(
        splits=preprocessed_bundle.splits,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )
    split_sizes = {
        split_name: int(split_data.inputs.shape[0])
        for split_name, split_data in preprocessed_bundle.splits.items()
    }
    logger.info("Window counts: %s", split_sizes)

    model_cls = MODEL_REGISTRY.get(config.model.name)
    if model_cls is None:
        raise ValueError(f"Unknown model: {config.model.name}")
    model = model_cls(
        input_dim=preprocessed_bundle.input_dim,
        horizon=config.window.horizon,
        input_kind=preprocessed_bundle.input_kind,
        model_config=config.model,
    )
    dummy_input = torch.zeros(
        (1, config.window.lookback, preprocessed_bundle.input_dim)
    )
    with torch.no_grad():
        model(dummy_input)

    logger.info(
        "Model parameters: %d",
        sum(parameter.numel() for parameter in model.parameters()),
    )

    trainer = Trainer(
        model=model, config=config.training, device=resolve_device(config.device)
    )
    training_result = trainer.fit(
        dataloaders["train"], dataloaders["val"], logger=logger
    )

    scaled_predictions = {}
    scaled_targets = {}
    for split_name in ("train", "val", "test"):
        scaled_predictions[split_name] = trainer.predict(dataloaders[split_name])
        scaled_targets[split_name] = trainer.collect_targets(dataloaders[split_name])

    evaluation_bundle = evaluate_predictions(
        scaled_predictions=scaled_predictions,
        scaled_targets=scaled_targets,
        scaler_bundle=windowed_bundle.scaler_bundle,
    )

    metrics_payload = {
        split_name: evaluation_bundle.splits[split_name].metrics
        for split_name in ("train", "val", "test")
    }
    prediction_rows = build_prediction_rows(
        split_name="test",
        split_data=preprocessed_bundle.splits["test"],
        predictions=evaluation_bundle.splits["test"].predictions,
        targets=evaluation_bundle.splits["test"].targets,
    )
    predictions_frame = pd.DataFrame(prediction_rows)
    metrics_frame = pd.DataFrame(
        [
            {"split": split_name, **metrics_payload[split_name]}
            for split_name in ("train", "val", "test")
        ]
    )

    save_yaml(experiment_config_to_dict(config), run_root / "resolved_config.yaml")
    save_json(metrics_payload, metrics_dir / "metrics.json")
    save_dataframe(metrics_frame, metrics_dir / "metrics.csv")
    save_dataframe(predictions_frame, run_root / "predictions.csv")
    save_joblib(
        {
            "target_scaler": windowed_bundle.scaler_bundle.target_scaler,
            "feature_scaler": windowed_bundle.scaler_bundle.feature_scaler,
            "scaler_name": windowed_bundle.scaler_bundle.scaler_name,
        },
        artifacts_dir / "scalers.joblib",
    )
    torch.save(
        {
            "model_state_dict": training_result.model.state_dict(),
            "model_name": config.model.name,
            "preprocess": preprocess_name,
            "input_dim": preprocessed_bundle.input_dim,
            "input_kind": preprocessed_bundle.input_kind,
            "horizon": config.window.horizon,
            "config": experiment_config_to_dict(config),
        },
        artifacts_dir / "best_model.pt",
    )

    plot_loss(training_result.history, plots_dir / "loss.png")
    plot_predictions(
        evaluation_bundle.splits["test"].predictions,
        evaluation_bundle.splits["test"].targets,
        plots_dir / "predictions.png",
    )
    plot_residuals(
        evaluation_bundle.splits["test"].predictions,
        evaluation_bundle.splits["test"].targets,
        plots_dir / "residuals.png",
    )
    write_run_summary(
        output_path=run_root / "summary.md",
        experiment_name=config.experiment_name,
        preprocess_name=preprocess_name,
        model_name=config.model.name,
        split_metrics=metrics_payload,
    )

    result = {
        "experiment_name": config.experiment_name,
        "preprocess": preprocess_name,
        "model_name": config.model.name,
        "seed": config.seed,
        "device": resolve_device(config.device),
        "train_MAE": metrics_payload["train"]["MAE"],
        "train_RMSE": metrics_payload["train"]["RMSE"],
        "train_MAPE": metrics_payload["train"]["MAPE"],
        "val_MAE": metrics_payload["val"]["MAE"],
        "val_RMSE": metrics_payload["val"]["RMSE"],
        "val_MAPE": metrics_payload["val"]["MAPE"],
        "test_MAE": metrics_payload["test"]["MAE"],
        "test_RMSE": metrics_payload["test"]["RMSE"],
        "test_MAPE": metrics_payload["test"]["MAPE"],
        "output_dir": str(run_root),
    }
    save_json(result, run_root / "run_summary.json")
    logger.info("Completed run for %s + %s", config.experiment_name, preprocess_name)
    return result


def write_batch_outputs(results: list[dict[str, Any]], output_root: str | Path) -> None:
    """Persist batch summary outputs after all six runs succeed."""
    summary_dir = ensure_dir(Path(output_root) / "summary")
    summary_frame = pd.DataFrame(results).sort_values(["experiment_name", "preprocess"])
    save_dataframe(summary_frame, summary_dir / "exp1_ab_results.csv")
    save_json(
        summary_frame.to_dict(orient="records"), summary_dir / "exp1_ab_results.json"
    )
    plot_batch_comparison(summary_frame, summary_dir / "exp1_ab_comparison.png")
    write_batch_summary(summary_dir / "exp1_ab_summary.md", results)
