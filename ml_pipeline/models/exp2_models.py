"""Pipeline adapters for Experiment 2 PyTorch models."""

from __future__ import annotations

from src.models.decoder_only_transformer import DecoderOnlyTransformerModel
from src.models.encoder_decoder_transformer import EncoderDecoderTransformerModel
from src.models.itransformer import ITransformerModel
from src.models.kernel_transformer import KernelTransformerModel
from src.utils.config import ModelConfig, TransformerConfig

from .exp1_cd_models import _TorchForecastingPipelineModel


def _base_transformer_config() -> TransformerConfig:
    """Shared transformer defaults for the Exp.2 variants."""
    cfg = TransformerConfig(
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.2,
        activation="gelu",
        pooling="mean",
        head_hidden_size=128,
        max_len=10000,
    )
    cfg.num_decoder_layers = None
    cfg.feature_dim = None
    cfg.fft_modes = None
    return cfg


def _default_exp2a_model_config() -> ModelConfig:
    return ModelConfig(
        name="decoder_only_transformer",
        transformer=_base_transformer_config(),
    )


def _default_exp2c_model_config() -> ModelConfig:
    return ModelConfig(
        name="encoder_decoder_transformer",
        transformer=_base_transformer_config(),
    )


def _default_exp2d_model_config() -> ModelConfig:
    cfg = _base_transformer_config()
    cfg.feature_dim = 64
    return ModelConfig(name="kernel_transformer", transformer=cfg)


def _default_exp2e_model_config() -> ModelConfig:
    return ModelConfig(
        name="itransformer",
        transformer=_base_transformer_config(),
    )


class DecoderOnlyTransformerPipelineModel(_TorchForecastingPipelineModel):
    """Exp.2.a Decoder-Only Transformer."""

    model_type = "exp2a_decoder_only_transformer"
    torch_model_cls = DecoderOnlyTransformerModel

    def __init__(self, *args, model_config: ModelConfig | None = None, **kwargs):
        super().__init__(
            *args,
            model_config=model_config or _default_exp2a_model_config(),
            **kwargs,
        )
class EncoderDecoderTransformerPipelineModel(_TorchForecastingPipelineModel):
    """Exp.2.c Encoder-Decoder Transformer."""

    model_type = "exp2c_encoder_decoder_transformer"
    torch_model_cls = EncoderDecoderTransformerModel

    def __init__(self, *args, model_config: ModelConfig | None = None, **kwargs):
        super().__init__(
            *args,
            model_config=model_config or _default_exp2c_model_config(),
            **kwargs,
        )


class KernelTransformerPipelineModel(_TorchForecastingPipelineModel):
    """Exp.2.d Kernel Transformer."""

    model_type = "exp2d_kernel_transformer"
    torch_model_cls = KernelTransformerModel

    def __init__(self, *args, model_config: ModelConfig | None = None, **kwargs):
        super().__init__(
            *args,
            model_config=model_config or _default_exp2d_model_config(),
            **kwargs,
        )


class ITransformerPipelineModel(_TorchForecastingPipelineModel):
    """Exp.2.e iTransformer."""

    model_type = "exp2e_itransformer"
    torch_model_cls = ITransformerModel

    def __init__(self, *args, model_config: ModelConfig | None = None, **kwargs):
        super().__init__(
            *args,
            model_config=model_config or _default_exp2e_model_config(),
            **kwargs,
        )
