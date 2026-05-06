"""Tests for YAML config loading and override behavior."""

from __future__ import annotations

import unittest
import uuid
from pathlib import Path
import shutil

from src.experiment import prepare_run_config
from src.utils.config import load_experiment_config

TEST_TMP_ROOT = Path("tests/.tmp")
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def make_test_dir() -> Path:
    """Create a writable test directory inside the repo."""
    path = TEST_TMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


class ConfigTests(unittest.TestCase):
    """Configuration loading tests."""

    def test_base_config_inheritance(self) -> None:
        config = load_experiment_config("configs/exp1a_cnn_bilstm.yaml")
        self.assertEqual(config.experiment_name, "exp1a_cnn_bilstm")
        self.assertEqual(config.window.lookback, 168)
        self.assertEqual(config.preprocessing.patch.patch_len, 24)
        self.assertEqual(config.model.name, "cnn_bilstm")
        self.assertIsNotNone(config.model.bilstm)

    def test_transformer_configs_load(self) -> None:
        exp1c_config = load_experiment_config("configs/exp1c_cnn_bilstm_transformer.yaml")
        self.assertEqual(exp1c_config.experiment_name, "exp1c_cnn_bilstm_transformer")
        self.assertEqual(exp1c_config.model.name, "cnn_bilstm_transformer")
        self.assertIsNotNone(exp1c_config.model.bilstm)
        self.assertIsNotNone(exp1c_config.model.transformer)
        self.assertIsNone(exp1c_config.model.xlstm)
        self.assertEqual(exp1c_config.model.transformer.d_model, 128)

        exp1d_config = load_experiment_config("configs/exp1d_cnn_transformer.yaml")
        self.assertEqual(exp1d_config.experiment_name, "exp1d_cnn_transformer")
        self.assertEqual(exp1d_config.model.name, "cnn_transformer")
        self.assertIsNone(exp1d_config.model.bilstm)
        self.assertIsNone(exp1d_config.model.xlstm)
        self.assertIsNotNone(exp1d_config.model.transformer)
        self.assertEqual(exp1d_config.window.horizon, 24)

    def test_cli_override_precedence(self) -> None:
        tmp_dir = make_test_dir()
        try:
            output_root = tmp_dir / "custom_outputs"
            config = prepare_run_config(
                config_path="configs/exp1b_cnn_xlstm.yaml",
                preprocess_name="patch",
                data_path="data/processed/final_dataset_full_clean.csv",
                device="cpu",
                output_root=str(output_root),
                seed=777,
            )
            self.assertTrue(config.data.path.endswith("final_dataset_full_clean.csv"))
            self.assertEqual(config.device, "cpu")
            self.assertEqual(config.outputs.root_dir, str(output_root))
            self.assertEqual(config.seed, 777)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
