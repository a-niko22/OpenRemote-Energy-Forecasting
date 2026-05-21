"""Shared ml_pipeline interface contracts."""

from .base_model import BaseModel
from .base_preprocessor import BasePreprocessor

__all__ = ["BaseModel", "BasePreprocessor"]