"""Model registry for Experiment 1."""

from src.models.cnn_bilstm import CNNBiLSTMModel
from src.models.cnn_bilstm_transformer import CNNBiLSTMTransformerModel
from src.models.cnn_transformer import CNNTransformerModel
from src.models.cnn_xlstm import CNNXLSTMModel

MODEL_REGISTRY = {
    "cnn_bilstm": CNNBiLSTMModel,
    "cnn_bilstm_transformer": CNNBiLSTMTransformerModel,
    "cnn_transformer": CNNTransformerModel,
    "cnn_xlstm": CNNXLSTMModel,
}
