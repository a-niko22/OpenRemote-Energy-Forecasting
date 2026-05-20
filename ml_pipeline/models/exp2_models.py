"""Pipeline adapters for Experiment 2 PyTorch models.

Four adapters covering Exp.2.a, 2.c, 2.d, 2.e. Exp.2.b (FFT) is dropped per
project decision — 2.b reuses DecoderOnlyTransformerPipelineModel paired with
the wavelet preprocessor at the Experiment composition level.

This file applies two workarounds for issues in src/ so we don't have to
touch the groupmates' code:

  1. src/models/layers.py defines SinusoidalPositionalEncoding twice. The
     second definition shadows the first and rejects the `dropout` kwarg that
     all four Exp2 models pass to it. We import the working first definition
     and rebind the module attribute before importing any Exp2 model.

  2. src/utils/config.TransformerConfig is missing three fields the Exp2
     models read (num_decoder_layers, feature_dim, fft_modes). We monkey-patch
     the dataclass to accept them with safe defaults.
"""

from __future__ import annotations

import dataclasses

# --- Workaround 1: restore the dropout-capable SinusoidalPositionalEncoding ---
# layers.py defines the class twice; class statements run top-to-bottom so the
# second (no-dropout) wins. Re-read the source, grab the FIRST class object,
# and rebind the module attribute before any Exp2 model imports the name.
import importlib
import inspect as _inspect

_layers = importlib.import_module("src.models.layers")
_source = _inspect.getsource(_layers)
_first_def_start = _source.index("class SinusoidalPositionalEncoding")
_second_def_start = _source.index("class SinusoidalPositionalEncoding", _first_def_start + 1)
_first_def_src = _source[_first_def_start:_second_def_start]

# Execute just the first class definition in a namespace that has layers.py's
# imports available, then rebind it on the module.
_ns: dict = {"nn": _layers.nn, "torch": _layers.torch, "math": _layers.math}
exec(_first_def_src, _ns)  # noqa: S102 — intentional, scoped to controlled source
_layers.SinusoidalPositionalEncoding = _ns["SinusoidalPositionalEncoding"]

# --- Workaround 2: add missing fields to TransformerConfig ---
from src.utils.config import TransformerConfig  # noqa: E402

if not hasattr(TransformerConfig, "num_decoder_layers"):
    # dataclasses.fields is frozen at class-creation time, so we patch __init__
    # by recreating the dataclass with extra fields. Simpler: just allow
    # attribute access by giving instances the missing attrs via __post_init__.
    _orig_init = TransformerConfig.__init__

    def _patched_init(self, *args, num_decoder_layers=None, feature_dim=None,
                      fft_modes=None, **kwargs):
        _orig_init(self, *args, **kwargs)
        object.__setattr__(self, "num_decoder_layers", num_decoder_layers)
        object.__setattr__(self, "feature_dim", feature_dim)
        object.__setattr__(self, "fft_modes", fft_modes)

    TransformerConfig.__init__ = _patched_init  # type: ignore[method-assign]
    TransformerConfig.num_decoder_layers = None  # type: ignore[attr-defined]
    TransformerConfig.feature_dim = None         # type: ignore[attr-defined]
    TransformerConfig.fft_modes = None           # type: ignore[attr-defined]

# --- Now safe to import Exp2 models ---
from src.models.decoder_only_transformer import DecoderOnlyTransformerModel  # noqa: E402
from src.models.encoder_decoder_transformer import EncoderDecoderTransformerModel  # noqa: E402
from src.models.itransformer import ITransformerModel  # noqa: E402
from src.models.kernel_transformer import KernelTransformerModel  # noqa: E402
from src.utils.config import ModelConfig  # noqa: E402

from .exp1_cd_models import _TorchForecastingPipelineModel  # noqa: E402


def _base_transformer_config() -> TransformerConfig:
    """Shared transformer defaults for the four Exp.2 variants."""
    cfg = TransformerConfig(
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.2,
        activation="gelu",
        pooling="mean",        # dataclass requires it; ignored by Exp2 models
        head_hidden_size=128,
        max_len=10000,
    )
    # Set the patched-in fields explicitly so the values are visible to
    # asdict() and get_config() logging.
    cfg.num_decoder_layers = None   # falls back to num_layers in 2a/2c
    cfg.feature_dim = None          # only used by 2d
    cfg.fft_modes = None
    return cfg


def _default_exp2a_model_config() -> ModelConfig:
    return ModelConfig(name="decoder_only_transformer", transformer=_base_transformer_config())


def _default_exp2c_model_config() -> ModelConfig:
    return ModelConfig(name="encoder_decoder_transformer", transformer=_base_transformer_config())


def _default_exp2d_model_config() -> ModelConfig:
    cfg = _base_transformer_config()
    cfg.feature_dim = 64   # Performer FAVOR+ random feature count
    return ModelConfig(name="kernel_transformer", transformer=cfg)


def _default_exp2e_model_config() -> ModelConfig:
    return ModelConfig(name="itransformer", transformer=_base_transformer_config())


class DecoderOnlyTransformerPipelineModel(_TorchForecastingPipelineModel):
    """Exp.2.a Decoder-Only Transformer. Also reused by 2.b with WaveletPreprocessor."""

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
    """Exp.2.d Kernel Transformer (Performer FAVOR+ linear attention)."""

    model_type = "exp2d_kernel_transformer"
    torch_model_cls = KernelTransformerModel

    def __init__(self, *args, model_config: ModelConfig | None = None, **kwargs):
        super().__init__(
            *args,
            model_config=model_config or _default_exp2d_model_config(),
            **kwargs,
        )


class ITransformerPipelineModel(_TorchForecastingPipelineModel):
    """Exp.2.e iTransformer (channels-as-tokens)."""

    model_type = "exp2e_itransformer"
    torch_model_cls = ITransformerModel

    def __init__(self, *args, model_config: ModelConfig | None = None, **kwargs):
        super().__init__(
            *args,
            model_config=model_config or _default_exp2e_model_config(),
            **kwargs,
        )