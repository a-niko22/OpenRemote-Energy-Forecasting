"""Luis-pipeline adapters for Experiment 1.a and 1.b PyTorch models."""

from __future__ import annotations

from src.models.cnn_bilstm import CNNBiLSTMModel
from src.models.cnn_xlstm import CNNXLSTMModel
from src.utils.config import BiLSTMConfig, CNNConfig, ModelConfig, XLSTMConfig

from .exp1_cd_models import _TorchForecastingPipelineModel


def _default_exp1a_model_config() -> ModelConfig:
    return ModelConfig(
        name="cnn_bilstm",
        patch_embed_dim=64,
        cnn=CNNConfig(
            conv_channels=[64, 128],
            kernel_size=3,
            use_pooling=False,
            pool_kernel=2,
            activation="relu",
            dropout=0.1,
        ),
        bilstm=BiLSTMConfig(
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            head_hidden_size=128,
        ),
    )


def _default_exp1b_model_config() -> ModelConfig:
    return ModelConfig(
        name="cnn_xlstm",
        patch_embed_dim=64,
        cnn=CNNConfig(
            conv_channels=[64, 128],
            kernel_size=3,
            use_pooling=False,
            pool_kernel=2,
            activation="relu",
            dropout=0.1,
        ),
        xlstm=XLSTMConfig(
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            projection_size=128,
            gate_clamp=5.0,
            stability_eps=1.0e-6,
            head_hidden_size=128,
        ),
    )


class CNNBiLSTMPipelineModel(_TorchForecastingPipelineModel):
    """Pipeline-compatible wrapper for Exp.1.a."""

    model_type = "exp1a_cnn_bilstm"
    torch_model_cls = CNNBiLSTMModel

    def __init__(self, *args, model_config: ModelConfig | None = None, **kwargs):
        super().__init__(
            *args,
            model_config=model_config or _default_exp1a_model_config(),
            **kwargs,
        )


class CNNXLSTMPipelineModel(_TorchForecastingPipelineModel):
    """Pipeline-compatible wrapper for Exp.1.b."""

    model_type = "exp1b_cnn_xlstm"
    torch_model_cls = CNNXLSTMModel

    def __init__(self, *args, model_config: ModelConfig | None = None, **kwargs):
        super().__init__(
            *args,
            model_config=model_config or _default_exp1b_model_config(),
            **kwargs,
        )
