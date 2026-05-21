"""Run a unified mock-data benchmark for Exp1, Exp2, and Prophet.

This script:
1) Checks/installs required dependencies,
2) Builds shared deterministic mock datasets,
3) Runs selected model variants with horizon forced to 48,
4) Runs Prophet through the shared ml_pipeline interface on paired mock tariff data,
5) Writes unified benchmark artifacts and dashboard/report outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ML_PIPELINE_ROOT = ROOT / "ml_pipeline"
if str(ML_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE_ROOT))

from src.data.dataset import load_time_series_data
from src.data.windowing import build_windowed_splits
from src.experiment import run_experiment
from src.utils.config import build_experiment_config, load_yaml_config
from src.utils.io import ensure_dir
from data.loader import chronological_split
from models.prophet_model import ProphetPipelineModel
from pipeline.experiment import Experiment as PipelineExperiment
from pipeline.experiment_runner import ExperimentRunner
from test_models_and_preprocessors.identity_preprocessor import IdentityPreprocessor


@dataclass(frozen=True)
class DeepRun:
    run_id: str
    family: str
    config_path: str
    preprocess: str
    label: str


EXP1_RUNS = [
    DeepRun("exp1a_norm", "exp1", "configs/exp1a_cnn_bilstm.yaml", "norm", "Exp.1.a CNN-BiLSTM + norm"),
    DeepRun("exp1a_wavelet", "exp1", "configs/exp1a_cnn_bilstm.yaml", "wavelet", "Exp.1.a CNN-BiLSTM + wavelet"),
    DeepRun("exp1a_patch", "exp1", "configs/exp1a_cnn_bilstm.yaml", "patch", "Exp.1.a CNN-BiLSTM + patch"),
    DeepRun("exp1b_norm", "exp1", "configs/exp1b_cnn_xlstm.yaml", "norm", "Exp.1.b CNN-xLSTM + norm"),
    DeepRun("exp1b_wavelet", "exp1", "configs/exp1b_cnn_xlstm.yaml", "wavelet", "Exp.1.b CNN-xLSTM + wavelet"),
    DeepRun("exp1b_patch", "exp1", "configs/exp1b_cnn_xlstm.yaml", "patch", "Exp.1.b CNN-xLSTM + patch"),
    DeepRun("exp1c_norm", "exp1", "configs/exp1c_cnn_bilstm_transformer.yaml", "norm", "Exp.1.c CNN-BiLSTM-Transformer + norm"),
    DeepRun("exp1c_wavelet", "exp1", "configs/exp1c_cnn_bilstm_transformer.yaml", "wavelet", "Exp.1.c CNN-BiLSTM-Transformer + wavelet"),
    DeepRun("exp1c_patch", "exp1", "configs/exp1c_cnn_bilstm_transformer.yaml", "patch", "Exp.1.c CNN-BiLSTM-Transformer + patch"),
    DeepRun("exp1d_norm", "exp1", "configs/exp1d_cnn_transformer.yaml", "norm", "Exp.1.d CNN-Transformer + norm"),
    DeepRun("exp1d_wavelet", "exp1", "configs/exp1d_cnn_transformer.yaml", "wavelet", "Exp.1.d CNN-Transformer + wavelet"),
    DeepRun("exp1d_patch", "exp1", "configs/exp1d_cnn_transformer.yaml", "patch", "Exp.1.d CNN-Transformer + patch"),
]

EXP2_RUNS = [
    DeepRun("exp2a", "exp2", "configs/exp2a_decoder_only.yaml", "exp2_standard", "Exp.2.a Decoder-only Transformer"),
    DeepRun("exp2b", "exp2", "configs/exp2b_fft_decoder.yaml", "exp2_fft", "Exp.2.b FFT Decoder Transformer"),
    DeepRun("exp2c", "exp2", "configs/exp2c_encoder_decoder.yaml", "exp2_standard", "Exp.2.c Encoder-Decoder Transformer"),
    DeepRun("exp2d", "exp2", "configs/exp2d_kernel.yaml", "exp2_kernel", "Exp.2.d Kernel Transformer"),
    DeepRun("exp2e", "exp2", "configs/exp2e_itransformer.yaml", "exp2_standard", "Exp.2.e iTransformer"),
]

PROPHET_RUN_ID = "prophet"
ALL_RUN_IDS = {run.run_id for run in EXP1_RUNS + EXP2_RUNS} | {PROPHET_RUN_ID}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all model families on shared mock benchmark data.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-root", default="outputs/mock_benchmark")
    parser.add_argument("--skip-exp2", action="store_true")
    parser.add_argument("--only", default=None, help="Comma-separated run ids. Include 'prophet' if needed.")
    parser.add_argument(
        "--skip-dep-install",
        action="store_true",
        help="Only check dependencies; do not auto-install missing packages.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def generate_base_tariff_data(days: int = 365, freq: str = "h") -> pd.DataFrame:
    """Mirror prophet/mock_data.py behavior without importing local prophet module."""
    np.random.seed(42)
    periods = days * 24
    timestamps = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=periods, freq=freq)

    hour = timestamps.hour
    daily = (
        -2 * np.cos(2 * np.pi * hour / 24)
        + 1.5 * np.cos(2 * np.pi * (hour - 8) / 12)
    )
    weekly = -0.5 * (timestamps.dayofweek >= 5).astype(float)
    day_of_year = timestamps.dayofyear
    seasonal = np.sin(2 * np.pi * day_of_year / 365)
    noise = np.random.normal(0, 0.3, periods)

    price = 0.15 + 0.05 * (daily + weekly + seasonal + noise)
    price = np.clip(price, 0.01, 0.50)
    return pd.DataFrame({"ds": timestamps, "y": price})


def module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def ensure_required_modules(selected_runs: list[DeepRun], run_prophet: bool, skip_install: bool) -> None:
    required = {"numpy": "numpy", "pandas": "pandas", "torch": "torch"}
    needs_exp2 = any(run.family == "exp2" for run in selected_runs)
    if needs_exp2:
        required["sklearn"] = "scikit-learn"
        required["feature_engine"] = "feature_engine"
        required["pywt"] = "PyWavelets"
    # Plotting fallback in src/evaluation/plots requires either matplotlib or PIL.
    has_matplotlib = module_exists("matplotlib")
    has_pil = module_exists("PIL")
    if not has_matplotlib and not has_pil:
        required["matplotlib"] = "matplotlib"

    missing = {module: pkg for module, pkg in required.items() if not module_exists(module)}
    if run_prophet and not module_exists("prophet"):
        missing["prophet"] = "prophet"
    if not missing:
        return
    if skip_install:
        raise RuntimeError(
            "Missing required packages: "
            + ", ".join(f"{mod} (pip: {pkg})" for mod, pkg in sorted(missing.items()))
        )

    packages = sorted(set(missing.values()))
    print("Installing missing packages:", ", ".join(packages))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])

    still_missing = [module for module in missing if not module_exists(module)]
    if still_missing:
        raise RuntimeError(f"Packages still missing after install: {still_missing}")


def select_runs(args: argparse.Namespace) -> tuple[list[DeepRun], bool]:
    deep_runs = EXP1_RUNS + ([] if args.skip_exp2 else EXP2_RUNS)
    requested_ids: set[str] | None = None
    if args.only:
        requested_ids = {token.strip() for token in args.only.split(",") if token.strip()}
        unknown = sorted(requested_ids - ALL_RUN_IDS)
        if unknown:
            raise ValueError(f"Unknown run ids in --only: {unknown}. Allowed: {sorted(ALL_RUN_IDS)}")

    if requested_ids is None:
        selected_deep = deep_runs
        run_prophet = True
    else:
        selected_deep = [run for run in deep_runs if run.run_id in requested_ids]
        run_prophet = PROPHET_RUN_ID in requested_ids
    return selected_deep, run_prophet


def build_mock_datasets(benchmark_dir: Path) -> tuple[Path, Path]:
    mock_dir = ensure_dir(benchmark_dir / "mock_data")
    base = generate_base_tariff_data(days=365, freq="h").sort_values("ds").reset_index(drop=True)

    ds = pd.to_datetime(base["ds"])
    y = base["y"].astype(float).to_numpy()
    t = np.arange(len(base), dtype=np.float64)

    hour = ds.dt.hour.to_numpy(dtype=np.float64)
    dow = ds.dt.dayofweek.to_numpy(dtype=np.float64)
    doy = ds.dt.dayofyear.to_numpy(dtype=np.float64)

    # Deterministic feature synthesis tied to the same target dynamics.
    temp = 10.0 + 9.0 * np.sin(2 * np.pi * doy / 365.0) + 2.0 * np.sin(2 * np.pi * hour / 24.0)
    cloud = np.clip(55.0 + 25.0 * np.sin(2 * np.pi * t / (24.0 * 10.0)) - 30.0 * y, 0.0, 100.0)
    wind = np.clip(6.0 + 1.5 * np.cos(2 * np.pi * t / (24.0 * 7.0)) + 2.0 * (1.0 - y), 0.2, None)
    radiation = np.clip(700.0 * np.sin(np.pi * hour / 24.0) * (1.0 - cloud / 120.0), 0.0, None)
    total_load = 980.0 + 85.0 * np.sin(2 * np.pi * hour / 24.0) + 45.0 * np.cos(2 * np.pi * dow / 7.0) + 140.0 * y
    generation = 640.0 + 0.45 * radiation + 15.0 * wind - 70.0 * (cloud / 100.0)
    gas_price = 27.0 + 2.5 * np.sin(2 * np.pi * doy / 365.0 + 0.7) + 8.0 * y
    demand_index = 1.0 + 0.05 * np.sin(2 * np.pi * t / (24.0 * 30.0)) + 0.02 * (total_load / np.mean(total_load) - 1.0)
    imbalance_signal = 0.3 * np.gradient(y) + 0.001 * np.gradient(total_load)

    exp_df = pd.DataFrame(
        {
            "time": ds,
            "price": y,
            "temperature_2m": temp.astype(np.float64),
            "cloud_cover": cloud.astype(np.float64),
            "wind_speed_10m": wind.astype(np.float64),
            "shortwave_radiation": radiation.astype(np.float64),
            "total_load": total_load.astype(np.float64),
            "generation_forecast": generation.astype(np.float64),
            "gas_price": gas_price.astype(np.float64),
            "demand_index": demand_index.astype(np.float64),
            "imbalance_signal": imbalance_signal.astype(np.float64),
        }
    )

    prophet_df = pd.DataFrame({"ds": ds, "y": y.astype(np.float64)})

    exp_path = mock_dir / "mock_energy_multivariate.csv"
    prophet_path = mock_dir / "mock_tariff_shared.csv"
    exp_df.to_csv(exp_path, index=False)
    prophet_df.to_csv(prophet_path, index=False)
    return exp_path, prophet_path


def make_temp_config(
    run: DeepRun,
    mock_dataset_path: Path,
    output_root: Path,
    device: str,
    temp_config_dir: Path,
) -> Path:
    payload = load_yaml_config(ROOT / run.config_path)
    payload["data"]["path"] = str(mock_dataset_path)
    payload["window"]["horizon"] = 48
    payload["device"] = device
    payload["outputs"]["root_dir"] = str(output_root)
    payload["experiment_name"] = f"{payload['experiment_name']}_mock48"

    ensure_dir(temp_config_dir)
    config_path = temp_config_dir / f"{run.run_id}.yaml"
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return config_path


def validate_non_empty_windows(config_path: Path) -> None:
    config = build_experiment_config(load_yaml_config(config_path))
    loaded = load_time_series_data(config.data)
    bundles = build_windowed_splits(loaded.frame, config)
    for split_name in ("train", "val", "test"):
        windows = bundles.splits[split_name].inputs.shape[0]
        if windows <= 0:
            raise ValueError(f"Zero windows generated for split '{split_name}' in config {config_path}.")


def run_deep_models(
    selected_runs: list[DeepRun],
    mock_dataset_path: Path,
    output_root: Path,
    device: str,
    benchmark_timestamp: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    temp_config_dir = ensure_dir(ROOT / "benchmarks" / "_tmp_mock_benchmark_configs")

    for run in selected_runs:
        config_path = make_temp_config(
            run=run,
            mock_dataset_path=mock_dataset_path,
            output_root=output_root,
            device=device,
            temp_config_dir=temp_config_dir,
        )
        validate_non_empty_windows(config_path)

        print(f"[RUN] {run.run_id}")
        started = time.perf_counter()
        result = run_experiment(config_path=config_path, preprocess_name=run.preprocess, device=device)
        duration = round(time.perf_counter() - started, 3)

        rows.append(
            {
                "run_id": run.run_id,
                "family": run.family,
                "model": result["experiment_name"],
                "variant": run.preprocess,
                "horizon_hours": 48,
                "heldout_MAE": result["test_MAE"],
                "heldout_RMSE": result["test_RMSE"],
                "heldout_MAPE": result["test_MAPE"],
                "coverage_pct": None,
                "cv_MAE": None,
                "cv_RMSE": None,
                "cv_MAPE": None,
                "cv_coverage_pct": None,
                "duration_sec": duration,
                "timestamp": benchmark_timestamp,
                "data_source": str(mock_dataset_path),
                "output_dir": result["output_dir"],
            }
        )
    return rows


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    value = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    return float(value)


def run_prophet_benchmark(mock_prophet_path: Path, benchmark_timestamp: str) -> dict[str, Any]:
    print("[RUN] prophet")
    started = time.perf_counter()
    frame = pd.read_csv(mock_prophet_path, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)
    y = frame["y"].astype(float).to_numpy()
    X = y[:, None]

    X_tr, y_tr, X_val, y_val, X_te, y_te = chronological_split(X, y, train_ratio=0.7, val_ratio=0.15)

    experiment = PipelineExperiment(
        "Prophet interface benchmark",
        IdentityPreprocessor(),
        ProphetPipelineModel(target_feature_index=0, freq="h"),
        input_len=168,
        horizon=48,
    )
    runner = ExperimentRunner(input_len=168, horizon=48)
    result = runner.run(
        experiment,
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        X_val=X_val,
        y_val=y_val,
    )

    heldout_mae = float(result["metrics"]["mae"])
    heldout_rmse = float(result["metrics"]["rmse"])
    heldout_mape = _mape(
        np.asarray(result["y_test"], dtype=np.float64),
        np.asarray(result["predictions"], dtype=np.float64),
    )
    duration = round(time.perf_counter() - started, 3)
    return {
        "run_id": "prophet",
        "family": "prophet",
        "model": str(result["model_config"].get("type", "prophet_interface_baseline")),
        "variant": "ml_pipeline_interface_48h",
        "horizon_hours": int(result["horizon"]),
        "heldout_MAE": heldout_mae,
        "heldout_RMSE": heldout_rmse,
        "heldout_MAPE": heldout_mape,
        "coverage_pct": None,
        "cv_MAE": None,
        "cv_RMSE": None,
        "cv_MAPE": None,
        "cv_coverage_pct": None,
        "duration_sec": duration,
        "timestamp": benchmark_timestamp,
        "data_source": str(mock_prophet_path),
        "output_dir": None,
    }


def write_svg(frame: pd.DataFrame, output_path: Path) -> None:
    # Deep-model view + all-model view (log) to keep Prophet visible.
    display_frame = frame.copy()
    display_frame["label"] = display_frame["run_id"]
    display_frame = display_frame.sort_values(["family", "run_id"])

    deep = display_frame[display_frame["family"] != "prophet"].copy()
    all_rows = display_frame.copy()

    width = 1500
    height = 860
    panel_h = 350
    margin = 26

    def _escape(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    def _panel(
        rows: pd.DataFrame,
        x: int,
        y: int,
        w: int,
        h: int,
        title: str,
        log_scale: bool,
    ) -> list[str]:
        lines: list[str] = []
        lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#fffdf8" stroke="#d8d0c4"/>')
        lines.append(f'<text x="{x + 14}" y="{y + 28}" font-size="18" font-weight="700" fill="#18201d">{_escape(title)}</text>')

        top = y + 46
        label_w = 200
        bar_w = w - label_w - 40
        n = max(len(rows), 1)
        row_h = (h - 66) / n

        maes = rows["heldout_MAE"].astype(float).tolist()
        rmses = rows["heldout_RMSE"].astype(float).tolist()

        def scale(value: float, values: list[float]) -> float:
            if not values:
                return 0.0
            if log_scale:
                logs = [math.log10(max(v, 1e-12)) for v in values]
                lo = min(logs)
                hi = max(logs)
                if hi == lo:
                    return 1.0
                return (math.log10(max(value, 1e-12)) - lo) / (hi - lo)
            lo = min(values)
            hi = max(values)
            if hi == lo:
                return 1.0
            return (value - lo) / (hi - lo)

        for idx, row in enumerate(rows.itertuples(index=False)):
            y0 = top + idx * row_h
            mae = float(row.heldout_MAE)
            rmse = float(row.heldout_RMSE)
            label = str(row.label)
            mae_len = 8 + (bar_w - 8) * scale(mae, maes)
            rmse_len = 8 + (bar_w - 8) * scale(rmse, rmses)
            lines.append(f'<text x="{x + 12}" y="{y0 + 14:.1f}" font-size="11" fill="#33403b">{_escape(label)}</text>')
            bx = x + label_w
            lines.append(f'<rect x="{bx}" y="{y0 + 2:.1f}" width="{mae_len:.1f}" height="8" rx="4" fill="#1f77b4"/>')
            lines.append(f'<rect x="{bx}" y="{y0 + 13:.1f}" width="{rmse_len:.1f}" height="8" rx="4" fill="#ff7f0e"/>')

        legend_y = y + h - 12
        lines.append(f'<rect x="{x + 14}" y="{legend_y - 10}" width="12" height="8" fill="#1f77b4"/>')
        lines.append(f'<text x="{x + 31}" y="{legend_y - 2}" font-size="11" fill="#33403b">MAE</text>')
        lines.append(f'<rect x="{x + 76}" y="{legend_y - 10}" width="12" height="8" fill="#ff7f0e"/>')
        lines.append(f'<text x="{x + 93}" y="{legend_y - 2}" font-size="11" fill="#33403b">RMSE</text>')
        scale_text = "log10 scale" if log_scale else "linear scale"
        lines.append(f'<text x="{x + w - 96}" y="{legend_y - 2}" font-size="11" fill="#64706b">{scale_text}</text>')
        return lines

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f6f3ee"/>',
        '<text x="26" y="42" font-size="28" font-weight="800" fill="#18201d">All Models Mock Benchmark</text>',
        '<text x="26" y="66" font-size="14" fill="#64706b">Shared mock data, 48h horizon, held-out metrics for all models. Prophet is run through the ml_pipeline interface.</text>',
    ]

    svg_lines.extend(_panel(deep, margin, 90, width - 2 * margin, panel_h, "Deep models only (Exp1 + Exp2)", log_scale=False))
    svg_lines.extend(_panel(all_rows, margin, 462, width - 2 * margin, panel_h, "All models including Prophet", log_scale=True))
    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines) + "\n", encoding="utf-8")


def build_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "# All Models Mock Benchmark",
        "",
        "This report compares Exp1, Exp2, and Prophet on shared synthetic mock data at 48h forecast horizon.",
        "",
        "## Figure",
        "",
        "![All models mock benchmark](./assets/all_models_mock_benchmark_comparison.svg)",
        "",
        "## Results",
        "",
        "| run_id | family | model | variant | heldout_MAE | heldout_RMSE | heldout_MAPE | coverage_pct | cv_MAE | cv_RMSE | cv_MAPE | cv_coverage_pct | duration_sec |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in frame.itertuples(index=False):
        def _fmt(value: Any, digits: int = 6) -> str:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return "-"
            if isinstance(value, (int, float, np.floating)):
                return f"{float(value):.{digits}f}"
            return str(value)

        lines.append(
            f"| `{row.run_id}` | `{row.family}` | `{row.model}` | `{row.variant}` | "
            f"{_fmt(row.heldout_MAE)} | {_fmt(row.heldout_RMSE)} | {_fmt(row.heldout_MAPE, 4)} | "
            f"{_fmt(row.coverage_pct, 2)} | {_fmt(row.cv_MAE)} | {_fmt(row.cv_RMSE)} | "
            f"{_fmt(row.cv_MAPE, 4)} | {_fmt(row.cv_coverage_pct, 2)} | {_fmt(row.duration_sec, 3)} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Deep models are reported on held-out test only.",
            "- Prophet row is produced through the ml_pipeline interface path and reports held-out metrics only.",
            "- Cross-validation and coverage columns are intentionally empty in this interface benchmark.",
            "- Mock data is deterministic and derived from a shared Prophet-style synthetic tariff series.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_dashboard(frame: pd.DataFrame, svg_rel_path: str) -> str:
    frame = frame.copy().sort_values(["family", "run_id"])
    rows_html = []
    mae_bars = []
    rmse_bars = []

    mae_max = max(float(v) for v in frame["heldout_MAE"]) if not frame.empty else 1.0
    rmse_max = max(float(v) for v in frame["heldout_RMSE"]) if not frame.empty else 1.0

    for row in frame.itertuples(index=False):
        family_label = str(row.family).upper()
        cov = "-" if pd.isna(row.coverage_pct) else f"{float(row.coverage_pct):.2f}"
        cv_mae = "-" if pd.isna(row.cv_MAE) else f"{float(row.cv_MAE):.6f}"
        cv_rmse = "-" if pd.isna(row.cv_RMSE) else f"{float(row.cv_RMSE):.6f}"
        cv_mape = "-" if pd.isna(row.cv_MAPE) else f"{float(row.cv_MAPE):.4f}"
        cv_cov = "-" if pd.isna(row.cv_coverage_pct) else f"{float(row.cv_coverage_pct):.2f}"
        rows_html.append(
            "<tr>"
            f"<td>{row.run_id}</td>"
            f"<td>{family_label}</td>"
            f"<td>{row.model}</td>"
            f"<td>{row.variant}</td>"
            f"<td class=\"number\">{float(row.heldout_MAE):.6f}</td>"
            f"<td class=\"number\">{float(row.heldout_RMSE):.6f}</td>"
            f"<td class=\"number\">{float(row.heldout_MAPE):.4f}</td>"
            f"<td class=\"number\">{cov}</td>"
            f"<td class=\"number\">{cv_mae}</td>"
            f"<td class=\"number\">{cv_rmse}</td>"
            f"<td class=\"number\">{cv_mape}</td>"
            f"<td class=\"number\">{cv_cov}</td>"
            f"<td class=\"number\">{float(row.duration_sec):.3f}</td>"
            "</tr>"
        )

        kind = "prophet" if row.family == "prophet" else "deep"
        mae_w = max(4.0, (float(row.heldout_MAE) / mae_max) * 100.0)
        rmse_w = max(4.0, (float(row.heldout_RMSE) / rmse_max) * 100.0)
        label = f"{row.run_id} ({family_label})"
        mae_bars.append(
            f"<div class=\"bar-row\"><span>{label}</span><div class=\"bar-bg\"><div class=\"bar {kind}\" style=\"width:{mae_w:.2f}%\"></div></div><strong>{float(row.heldout_MAE):.6f}</strong></div>"
        )
        rmse_bars.append(
            f"<div class=\"bar-row\"><span>{label}</span><div class=\"bar-bg\"><div class=\"bar {kind}\" style=\"width:{rmse_w:.2f}%\"></div></div><strong>{float(row.heldout_RMSE):.6f}</strong></div>"
        )

    exp1_count = int((frame["family"] == "exp1").sum())
    exp2_count = int((frame["family"] == "exp2").sum())
    prophet_count = int((frame["family"] == "prophet").sum())
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>All Models Mock Benchmark</title>
  <style>
    :root {{
      --bg: #f6f3ee;
      --ink: #18201d;
      --muted: #64706b;
      --line: #d8d0c4;
      --panel: #fffdf8;
      --accent: #0f766e;
      --accent-2: #b45309;
      --shadow: 0 18px 45px rgba(24, 32, 29, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Aptos", "Segoe UI", Tahoma, sans-serif;
      line-height: 1.5;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: linear-gradient(120deg, #fffdf8 0%, #edf4ef 54%, #f7eadb 100%);
    }}
    .wrap {{
      width: min(1260px, calc(100% - 32px));
      margin: 0 auto;
    }}
    .hero {{
      padding: 44px 0 32px;
    }}
    .eyebrow {{
      margin: 0 0 10px;
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(32px, 5vw, 54px);
      line-height: 1.04;
    }}
    .subtitle {{
      max-width: 920px;
      margin: 14px 0 0;
      color: var(--muted);
      font-size: 17px;
    }}
    .stats {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      margin-top: 24px;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel);
      box-shadow: var(--shadow);
      padding: 18px;
      margin-top: 18px;
    }}
    .metric-label {{
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .metric-value {{
      margin: 0;
      font-size: 30px;
      font-weight: 800;
      line-height: 1;
    }}
    .metric-note {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 220px 1fr 92px;
      gap: 12px;
      align-items: center;
      margin: 10px 0;
      font-size: 14px;
    }}
    .bar-bg {{
      height: 13px;
      border-radius: 999px;
      background: #e4ded4;
      overflow: hidden;
    }}
    .bar {{
      height: 100%;
      border-radius: inherit;
      background: var(--accent);
    }}
    .bar.prophet {{
      background: var(--accent-2);
    }}
    .chart-note {{
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: var(--panel);
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: middle;
      font-size: 13px;
    }}
    th {{
      background: #ebe5da;
      font-size: 11px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .number {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    .note {{
      border-left: 4px solid var(--accent-2);
      background: #fff8ec;
      padding: 14px 16px;
      color: #43362b;
      border-radius: 6px;
      margin-top: 14px;
    }}
    main {{
      padding: 24px 0 56px;
    }}
    @media (max-width: 920px) {{
      .stats {{ grid-template-columns: 1fr; }}
      .bar-row {{
        grid-template-columns: 1fr;
        gap: 6px;
      }}
      .number {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap hero">
      <p class="eyebrow">Unified Benchmark</p>
      <h1>Exp1, Exp2, and Prophet on Shared Mock Data</h1>
      <p class="subtitle">
        Full-config benchmark with horizon aligned to 48 hours. All model families are evaluated on held-out windows; Prophet is executed through the shared ml_pipeline interface contract.
      </p>
      <section class="stats">
        <article class="panel">
          <p class="metric-label">Total Runs</p>
          <p class="metric-value">{len(frame)}</p>
          <p class="metric-note">All selected benchmark variants</p>
        </article>
        <article class="panel">
          <p class="metric-label">Exp1 / Exp2 Runs</p>
          <p class="metric-value">{exp1_count} / {exp2_count}</p>
          <p class="metric-note">Deep-learning held-out metrics</p>
        </article>
        <article class="panel">
          <p class="metric-label">Prophet Rows</p>
          <p class="metric-value">{prophet_count}</p>
          <p class="metric-note">Held-out metrics via interface path</p>
        </article>
      </section>
    </div>
  </header>
  <main class="wrap">
    <section class="panel">
      <h2 style="margin: 0 0 10px;">Comparison Figure</h2>
      <img src="{svg_rel_path}" alt="All models mock benchmark comparison">
      <p class="chart-note">The lower panel in the figure uses log scale so Prophet and deep model ranges are visible together.</p>
    </section>
    <section class="panel">
      <h2 style="margin: 0 0 10px;">Held-out MAE (all runs)</h2>
      {''.join(mae_bars)}
      <p class="chart-note">Bar widths are relative within this panel.</p>
    </section>
    <section class="panel">
      <h2 style="margin: 0 0 10px;">Held-out RMSE (all runs)</h2>
      {''.join(rmse_bars)}
      <p class="chart-note">Bar widths are relative within this panel.</p>
    </section>
    <section class="panel">
      <h2 style="margin: 0 0 10px;">Final Metrics Table</h2>
      <table>
        <thead>
          <tr>
            <th>run_id</th>
            <th>family</th>
            <th>model</th>
            <th>variant</th>
            <th class="number">heldout_MAE</th>
            <th class="number">heldout_RMSE</th>
            <th class="number">heldout_MAPE</th>
            <th class="number">coverage_pct</th>
            <th class="number">cv_MAE</th>
            <th class="number">cv_RMSE</th>
            <th class="number">cv_MAPE</th>
            <th class="number">cv_coverage_pct</th>
            <th class="number">duration_sec</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
      <div class="note">
        Shared synthetic data is used for all families here by design. Prophet now runs through the same ml_pipeline experiment contract used by wrapper models, but model internals still differ by architecture.
      </div>
    </section>
  </main>
</body>
</html>
"""


def persist_outputs(frame: pd.DataFrame) -> None:
    benchmark_dir = ensure_dir(ROOT / "benchmarks")
    reports_dir = ensure_dir(ROOT / "reports")
    assets_dir = ensure_dir(reports_dir / "assets")

    csv_path = benchmark_dir / "all_models_mock_benchmark.csv"
    json_path = benchmark_dir / "all_models_mock_benchmark.json"
    svg_path = assets_dir / "all_models_mock_benchmark_comparison.svg"
    md_path = reports_dir / "all_models_mock_benchmark.md"
    html_path = reports_dir / "all_models_mock_benchmark_dashboard.html"

    frame_sorted = frame.sort_values(["family", "run_id"]).reset_index(drop=True)
    frame_sorted.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(frame_sorted.to_dict(orient="records"), indent=2, default=str),
        encoding="utf-8",
    )
    write_svg(frame_sorted, svg_path)
    md_path.write_text(build_markdown(frame_sorted), encoding="utf-8")
    html_path.write_text(
        build_dashboard(
            frame_sorted,
            "./assets/all_models_mock_benchmark_comparison.svg",
        ),
        encoding="utf-8",
    )

    print("Wrote:", csv_path)
    print("Wrote:", json_path)
    print("Wrote:", svg_path)
    print("Wrote:", md_path)
    print("Wrote:", html_path)


def main() -> None:
    args = parse_args()
    selected_runs, run_prophet_flag = select_runs(args)
    if not selected_runs and not run_prophet_flag:
        raise ValueError("No runs selected. Use default selection or include valid ids in --only.")

    ensure_required_modules(
        selected_runs=selected_runs,
        run_prophet=run_prophet_flag,
        skip_install=args.skip_dep_install,
    )

    benchmark_timestamp = utc_now_iso()
    benchmark_dir = ensure_dir(ROOT / "benchmarks")
    output_root = ensure_dir(Path(args.output_root))
    mock_exp_path, mock_prophet_path = build_mock_datasets(benchmark_dir)

    all_rows: list[dict[str, Any]] = []
    if selected_runs:
        all_rows.extend(
            run_deep_models(
                selected_runs=selected_runs,
                mock_dataset_path=mock_exp_path,
                output_root=output_root,
                device=args.device,
                benchmark_timestamp=benchmark_timestamp,
            )
        )
    if run_prophet_flag:
        all_rows.append(run_prophet_benchmark(mock_prophet_path, benchmark_timestamp))

    if not all_rows:
        raise RuntimeError("No benchmark rows were produced.")
    persist_outputs(pd.DataFrame(all_rows))


if __name__ == "__main__":
    main()
