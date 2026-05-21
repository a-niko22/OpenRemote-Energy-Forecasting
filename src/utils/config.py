"""Typed configuration loading with lightweight YAML inheritance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import json

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


@dataclass
class DataConfig:
    path: str
    timestamp_col: str
    target_col: str
    feature_cols: list[str] | None = None
    resample_freq: str | None = None


@dataclass
class WindowConfig:
    lookback: int
    horizon: int
    stride: int


@dataclass
class SplitConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float


@dataclass
class NormPreprocessingConfig:
    method: str = "standard"


@dataclass
class WaveletPreprocessingConfig:
    wavelet_name: str = "db1"
    level: int = 1
    mode: str = "concat"


@dataclass
class PatchPreprocessingConfig:
    patch_len: int = 24
    patch_stride: int = 12


@dataclass
class PreprocessingConfig:
    norm: NormPreprocessingConfig = field(default_factory=NormPreprocessingConfig)
    wavelet: WaveletPreprocessingConfig = field(default_factory=WaveletPreprocessingConfig)
    patch: PatchPreprocessingConfig = field(default_factory=PatchPreprocessingConfig)


@dataclass
class CNNConfig:
    conv_channels: list[int]
    kernel_size: int
    use_pooling: bool
    pool_kernel: int
    activation: str
    dropout: float


@dataclass
class BiLSTMConfig:
    hidden_size: int
    num_layers: int
    dropout: float
    head_hidden_size: int


@dataclass
class XLSTMConfig:
    hidden_size: int
    num_layers: int
    dropout: float
    projection_size: int
    gate_clamp: float
    stability_eps: float
    head_hidden_size: int


@dataclass
class TransformerConfig:
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    activation: str = "relu"
    pooling: str = "mean"
    head_hidden_size: int = 128
    max_len: int = 10000
    num_decoder_layers: int | None = None
    fft_modes: int | None = None
    fft_kind: str | None = None
    kernel_type: str | None = None
    feature_dim: int | None = None


@dataclass
class ModelConfig:
    name: str
    patch_embed_dim: int = 64
    cnn: CNNConfig | None = None
    bilstm: BiLSTMConfig | None = None
    xlstm: XLSTMConfig | None = None
    transformer: TransformerConfig | None = None


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    patience: int
    lr_scheduler_patience: int
    lr_scheduler_factor: float
    grad_clip: float
    num_workers: int


@dataclass
class OutputConfig:
    root_dir: str


@dataclass
class ExperimentConfig:
    experiment_name: str
    seed: int
    device: str
    data: DataConfig
    window: WindowConfig
    split: SplitConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig
    outputs: OutputConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    raw_text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(raw_text) or {}
    else:
        data = json.loads(raw_text)
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must deserialize to a mapping: {path}")
    return data


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file with optional relative base inheritance."""
    config_path = Path(path)
    raw_config = _read_yaml(config_path)
    base_name = raw_config.pop("base_config", None)
    if base_name:
        base_path = (config_path.parent / base_name).resolve()
        base_config = load_yaml_config(base_path)
        return _deep_merge(base_config, raw_config)
    return raw_config


def _build_data_config(data: dict[str, Any]) -> DataConfig:
    return DataConfig(
        path=data["path"],
        timestamp_col=data["timestamp_col"],
        target_col=data["target_col"],
        feature_cols=data.get("feature_cols"),
        resample_freq=data.get("resample_freq"),
    )


def _build_window_config(data: dict[str, Any]) -> WindowConfig:
    return WindowConfig(
        lookback=int(data["lookback"]),
        horizon=int(data["horizon"]),
        stride=int(data["stride"]),
    )


def _build_split_config(data: dict[str, Any]) -> SplitConfig:
    return SplitConfig(
        train_ratio=float(data["train_ratio"]),
        val_ratio=float(data["val_ratio"]),
        test_ratio=float(data["test_ratio"]),
    )


def _build_preprocessing_config(data: dict[str, Any]) -> PreprocessingConfig:
    return PreprocessingConfig(
        norm=NormPreprocessingConfig(**data.get("norm", {})),
        wavelet=WaveletPreprocessingConfig(**data.get("wavelet", {})),
        patch=PatchPreprocessingConfig(**data.get("patch", {})),
    )


def _build_model_config(data: dict[str, Any]) -> ModelConfig:
    cnn_data = data.get("cnn")
    bilstm_data = data.get("bilstm")
    xlstm_data = data.get("xlstm")
    transformer_data = data.get("transformer")
    if transformer_data:
        transformer_data = dict(transformer_data)
        if "pool" in transformer_data and "pooling" not in transformer_data:
            transformer_data["pooling"] = transformer_data.pop("pool")
    return ModelConfig(
        name=data["name"],
        patch_embed_dim=int(data.get("patch_embed_dim", 64)),
        cnn=CNNConfig(**cnn_data) if cnn_data else None,
        bilstm=BiLSTMConfig(**bilstm_data) if bilstm_data else None,
        xlstm=XLSTMConfig(**xlstm_data) if xlstm_data else None,
        transformer=TransformerConfig(**transformer_data) if transformer_data else None,
    )


def _build_training_config(data: dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(
        batch_size=int(data["batch_size"]),
        epochs=int(data["epochs"]),
        lr=float(data["lr"]),
        weight_decay=float(data["weight_decay"]),
        patience=int(data["patience"]),
        lr_scheduler_patience=int(data["lr_scheduler_patience"]),
        lr_scheduler_factor=float(data["lr_scheduler_factor"]),
        grad_clip=float(data["grad_clip"]),
        num_workers=int(data["num_workers"]),
    )


def _build_output_config(data: dict[str, Any]) -> OutputConfig:
    return OutputConfig(root_dir=data["root_dir"])


def build_experiment_config(config_data: dict[str, Any]) -> ExperimentConfig:
    """Construct the typed experiment config."""
    return ExperimentConfig(
        experiment_name=config_data["experiment_name"],
        seed=int(config_data["seed"]),
        device=config_data["device"],
        data=_build_data_config(config_data["data"]),
        window=_build_window_config(config_data["window"]),
        split=_build_split_config(config_data["split"]),
        preprocessing=_build_preprocessing_config(config_data["preprocessing"]),
        model=_build_model_config(config_data["model"]),
        training=_build_training_config(config_data["training"]),
        outputs=_build_output_config(config_data["outputs"]),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and type-check an experiment config file."""
    return build_experiment_config(load_yaml_config(path))


def experiment_config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert a typed config into a plain serializable mapping."""
    if is_dataclass(config):
        return asdict(config)
    raise TypeError("Expected a dataclass-backed ExperimentConfig.")
