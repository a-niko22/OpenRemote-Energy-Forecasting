"""Pipeline-compatible model adapters."""

# Ensure repo root is on sys.path so `import src.models...` works
# when this package is imported from inside ml_pipeline/.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .exp1_ab_models import (
    CNNBiLSTMPipelineModel,
    CNNXLSTMPipelineModel,
)
from .exp1_cd_models import (
    CNNBiLSTMTransformerPipelineModel,
    CNNTransformerPipelineModel,
)
from .exp2_models import (
    DecoderOnlyTransformerPipelineModel,
    EncoderDecoderTransformerPipelineModel,
    FFTDecoderTransformerPipelineModel,
    KernelTransformerPipelineModel,
    ITransformerPipelineModel,
)

__all__ = [
    "CNNBiLSTMPipelineModel",
    "CNNXLSTMPipelineModel",
    "CNNBiLSTMTransformerPipelineModel",
    "CNNTransformerPipelineModel",
    "DecoderOnlyTransformerPipelineModel",
    "EncoderDecoderTransformerPipelineModel",
    "FFTDecoderTransformerPipelineModel",
    "KernelTransformerPipelineModel",
    "ITransformerPipelineModel",
]
