"""Model registry for all experiments."""

from src.models.cnn_bilstm import CNNBiLSTMModel
from src.models.cnn_bilstm_transformer import CNNBiLSTMTransformerModel
from src.models.cnn_transformer import CNNTransformerModel
from src.models.cnn_xlstm import CNNXLSTMModel
from src.models.decoder_only_transformer import DecoderOnlyTransformerModel
from src.models.fft_decoder_transformer import FFTDecoderTransformerModel
from src.models.encoder_decoder_transformer import EncoderDecoderTransformerModel
from src.models.kernel_transformer import KernelTransformerModel
from src.models.itransformer import ITransformerModel

MODEL_REGISTRY = {
    "cnn_bilstm": CNNBiLSTMModel,
    "cnn_bilstm_transformer": CNNBiLSTMTransformerModel,
    "cnn_transformer": CNNTransformerModel,
    "cnn_xlstm": CNNXLSTMModel,
    "decoder_only_transformer": DecoderOnlyTransformerModel,
    "fft_decoder_transformer": FFTDecoderTransformerModel,
    "encoder_decoder_transformer": EncoderDecoderTransformerModel,
    "kernel_transformer": KernelTransformerModel,
    "itransformer": ITransformerModel,
}
