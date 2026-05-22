"""Pipeline-compatible model adapters.

Adapters are loaded lazily so Prophet-only and baseline runs do not import the
torch-backed experiment families until those classes are actually requested.
"""

from __future__ import annotations

from importlib import import_module
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_EXPORTS = {
    "CNNBiLSTMPipelineModel": "exp1_ab_models",
    "CNNXLSTMPipelineModel": "exp1_ab_models",
    "CNNBiLSTMTransformerPipelineModel": "exp1_cd_models",
    "CNNTransformerPipelineModel": "exp1_cd_models",
    "DecoderOnlyTransformerPipelineModel": "exp2_models",
    "EncoderDecoderTransformerPipelineModel": "exp2_models",
    "KernelTransformerPipelineModel": "exp2_models",
    "ITransformerPipelineModel": "exp2_models",
    "ProphetPipelineModel": "prophet_model",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{_EXPORTS[name]}")
    value = getattr(module, name)
    globals()[name] = value
    return value