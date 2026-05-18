"""Smoke tests for end-to-end runs on a tiny synthetic dataset."""

from __future__ import annotations

import unittest
import uuid
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from src.experiment import run_experiment, write_batch_outputs

TEST_TMP_ROOT = Path("tests/.tmp")
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def make_test_dir() -> Path:
    """Create a writable test directory inside the repo."""
    path = TEST_TMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def _make_tiny_dataset(path: Path) -> None:
    rows = 260
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=rows, freq="h"),
            "price": 30.0 + np.sin(np.arange(rows) / 6.0) * 5.0,
            "load": 1000.0 + np.cos(np.arange(rows) / 5.0) * 20.0,
            "temp": 10.0 + np.sin(np.arange(rows) / 4.0),
        }
    )
    frame.to_csv(path, index=False)


class SmokePipelineTests(unittest.TestCase):
    """Smoke tests for training and batch outputs."""

    def test_smoke_training_for_both_architectures(self) -> None:
        tmp_path = make_test_dir()
        try:
            dataset_path = tmp_path / "tiny.csv"
            _make_tiny_dataset(dataset_path)
            output_root = tmp_path / "outputs"

            bilstm_result = run_experiment(
                config_path="configs/exp1a_cnn_bilstm.yaml",
                preprocess_name="norm",
                data_path=str(dataset_path),
                device="cpu",
                output_root=str(output_root),
                seed=123,
            )
            xlstm_result = run_experiment(
                config_path="configs/exp1b_cnn_xlstm.yaml",
                preprocess_name="patch",
                data_path=str(dataset_path),
                device="cpu",
                output_root=str(output_root),
                seed=123,
            )

            for result in (bilstm_result, xlstm_result):
                run_dir = Path(result["output_dir"])
                self.assertTrue((run_dir / "artifacts" / "best_model.pt").exists())
                self.assertTrue((run_dir / "metrics" / "metrics.json").exists())
                self.assertTrue((run_dir / "plots" / "loss.png").exists())
                self.assertTrue((run_dir / "predictions.csv").exists())
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_batch_summary_writer_creates_expected_outputs(self) -> None:
        tmp_dir = make_test_dir()
        try:
            output_root = tmp_dir / "outputs"
            results = [
                {
                    "experiment_name": "exp1a_cnn_bilstm",
                    "preprocess": "norm",
                    "test_MAE": 1.0,
                    "test_RMSE": 2.0,
                    "test_MAPE": 3.0,
                },
                {
                    "experiment_name": "exp1a_cnn_bilstm",
                    "preprocess": "wavelet",
                    "test_MAE": 1.1,
                    "test_RMSE": 2.1,
                    "test_MAPE": 3.1,
                },
                {
                    "experiment_name": "exp1a_cnn_bilstm",
                    "preprocess": "patch",
                    "test_MAE": 1.2,
                    "test_RMSE": 2.2,
                    "test_MAPE": 3.2,
                },
                {
                    "experiment_name": "exp1b_cnn_xlstm",
                    "preprocess": "norm",
                    "test_MAE": 1.3,
                    "test_RMSE": 2.3,
                    "test_MAPE": 3.3,
                },
                {
                    "experiment_name": "exp1b_cnn_xlstm",
                    "preprocess": "wavelet",
                    "test_MAE": 1.4,
                    "test_RMSE": 2.4,
                    "test_MAPE": 3.4,
                },
                {
                    "experiment_name": "exp1b_cnn_xlstm",
                    "preprocess": "patch",
                    "test_MAE": 1.5,
                    "test_RMSE": 2.5,
                    "test_MAPE": 3.5,
                },
            ]
            write_batch_outputs(results, output_root=output_root)

            summary_dir = output_root / "summary"
            self.assertTrue((summary_dir / "exp1_ab_results.csv").exists())
            self.assertTrue((summary_dir / "exp1_ab_results.json").exists())
            self.assertTrue((summary_dir / "exp1_ab_comparison.png").exists())
            self.assertTrue((summary_dir / "exp1_ab_summary.md").exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
