"""Model registry for all experiments."""

from src.models.cnn_bilstm import CNNBiLSTMModel
from src.models.cnn_xlstm import CNNXLSTMModel
from src.models.vanilla_transformer import VanillaTransformerModel
from src.models.itransformer import ITransformerModel

MODEL_REGISTRY = {
    "cnn_bilstm": CNNBiLSTMModel,
    "cnn_xlstm": CNNXLSTMModel,
    "vanilla_transformer": VanillaTransformerModel,
    "itransformer": ITransformerModel,
}
