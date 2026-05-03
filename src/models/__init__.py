"""Model registry for Experiment 1."""

from src.models.cnn_bilstm import CNNBiLSTMModel
from src.models.cnn_xlstm import CNNXLSTMModel

MODEL_REGISTRY = {
    "cnn_bilstm": CNNBiLSTMModel,
    "cnn_xlstm": CNNXLSTMModel,
}
