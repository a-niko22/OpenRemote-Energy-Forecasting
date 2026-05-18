"""Pipeline-compatible model adapters."""

from .exp1_cd_models import (
    CNNBiLSTMTransformerPipelineModel,
    CNNTransformerPipelineModel,
)

__all__ = [
    "CNNBiLSTMTransformerPipelineModel",
    "CNNTransformerPipelineModel",
]
