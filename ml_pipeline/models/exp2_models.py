"""ml_pipeline adapters for Experiment 2 PyTorch models (2.a – 2.e)."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Type

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
ML_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(ML_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE_ROOT))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from interfaces.base_model import BaseModel
except ModuleNotFoundError:
    sys.modules.pop("interfaces", None)
    from interfaces.base_model import BaseModel

from src.models.decoder_only_transformer import DecoderOnlyTransformerModel
from src.models.fft_decoder_transformer import FFTDecoderTransformerModel
from src.models.encoder_decoder_transformer import EncoderDecoderTransformerModel
from src.models.kernel_transformer import KernelTransformerModel
from src.models.itransformer import ITransformerModel


# ---------------------------------------------------------------------------
# Self-contained config objects
# src/utils/config.py::TransformerConfig lacks the exp2-specific fields
# (num_decoder_layers, fft_modes, feature_dim, etc.), so adapters build their
# own lightweight config whose attributes the exp2 models read via attribute
# access only — never asdict / dataclass introspection.
# ---------------------------------------------------------------------------

class _Exp2TransformerCfg:
    """Namespace exposing every transformer attribute exp2 models may read."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        head_hidden_size: int,
        num_decoder_layers: int | None = None,
        fft_modes: int | None = None,
        feature_dim: int | None = None,
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.head_hidden_size = head_hidden_size
        self.num_decoder_layers = num_decoder_layers
        self.fft_modes = fft_modes
        self.feature_dim = feature_dim

    def as_dict(self) -> dict:
        return vars(self)


class _Exp2ModelCfg:
    """Minimal ModelConfig stand-in — exp2 models only access .transformer."""

    def __init__(self, name: str, transformer: _Exp2TransformerCfg):
        self.name = name
        self.transformer = transformer
        # Other ModelConfig fields set to None so isinstance checks in
        # src/ code that may branch on them don't raise AttributeError.
        self.cnn = None
        self.bilstm = None
        self.xlstm = None
        self.patch_embed_dim = 64


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class _Exp2TorchPipelineModel(BaseModel):
    """BaseModel adapter wrapping an Experiment 2 torch module.

    Subclasses set torch_model_cls and provide default transformer hyperparams
    via _default_transformer_kwargs. All hyperparams are overridable at
    construction time.
    """

    model_type: str = "exp2_torch"
    torch_model_cls: Type[nn.Module]

    # Subclasses override to supply per-experiment defaults.
    _default_transformer_kwargs: dict[str, Any] = {}

    def __init__(
        self,
        # Transformer hyperparams — defaults come from the subclass.
        d_model: int | None = None,
        nhead: int | None = None,
        num_layers: int | None = None,
        dim_feedforward: int | None = None,
        dropout: float | None = None,
        head_hidden_size: int | None = None,
        num_decoder_layers: int | None = None,
        fft_modes: int | None = None,
        feature_dim: int | None = None,
        # Training hyperparams.
        device: str | torch.device = "auto",
        default_epochs: int = 10,
        default_batch_size: int = 32,
        default_learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        loss: str = "mse",
        optimizer: str = "adam",
        seed: int | None = 42,
    ):
        defaults = self._default_transformer_kwargs

        self._transformer_cfg = _Exp2TransformerCfg(
            d_model=d_model if d_model is not None else defaults["d_model"],
            nhead=nhead if nhead is not None else defaults["nhead"],
            num_layers=num_layers if num_layers is not None else defaults["num_layers"],
            dim_feedforward=dim_feedforward if dim_feedforward is not None else defaults["dim_feedforward"],
            dropout=dropout if dropout is not None else defaults["dropout"],
            head_hidden_size=head_hidden_size if head_hidden_size is not None else defaults["head_hidden_size"],
            num_decoder_layers=num_decoder_layers if num_decoder_layers is not None else defaults.get("num_decoder_layers"),
            fft_modes=fft_modes if fft_modes is not None else defaults.get("fft_modes"),
            feature_dim=feature_dim if feature_dim is not None else defaults.get("feature_dim"),
        )
        self.model_config = _Exp2ModelCfg(
            name=self.model_type,
            transformer=self._transformer_cfg,
        )

        self.input_kind = "sequence"  # all exp2 models use sequence; they ignore the arg
        self.requested_device = str(device)
        self.device = self._resolve_device(device)
        self.default_epochs = int(default_epochs)
        self.default_batch_size = int(default_batch_size)
        self.default_learning_rate = float(default_learning_rate)
        self.weight_decay = float(weight_decay)
        self.loss = loss
        self.optimizer = optimizer
        self.seed = seed

        self.model: nn.Module | None = None
        self.input_dim: int | None = None
        self.horizon: int | None = None
        self.is_fitted = False
        self.training_history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if str(device) == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _ensure_3d_features(X) -> np.ndarray:
        X_array = np.asarray(X, dtype=np.float32)
        if X_array.ndim == 2:
            return X_array[:, None, :]
        if X_array.ndim != 3:
            raise ValueError(
                "X must have shape (n_windows, input_len, n_features) "
                "or (n_windows, n_features)."
            )
        return X_array

    @staticmethod
    def _ensure_2d_targets(y) -> np.ndarray:
        y_array = np.asarray(y, dtype=np.float32)
        if y_array.ndim == 1:
            return y_array[:, None]
        if y_array.ndim != 2:
            raise ValueError("y must have shape (n_windows,) or (n_windows, horizon).")
        return y_array

    def _build_model(self, input_dim: int, horizon: int) -> nn.Module:
        return self.torch_model_cls(
            input_dim=input_dim,
            horizon=horizon,
            input_kind=self.input_kind,
            model_config=self.model_config,
        ).to(self.device)

    def _build_loss(self) -> nn.Module:
        if self.loss.lower() in {"mae", "l1", "l1loss"}:
            return nn.L1Loss()
        if self.loss.lower() in {"mse", "mseloss"}:
            return nn.MSELoss()
        raise ValueError(f"Unsupported loss: {self.loss}")

    def _build_optimizer(self, learning_rate: float) -> torch.optim.Optimizer:
        if self.model is None:
            raise RuntimeError("Model has not been initialized.")
        opt_name = self.optimizer.lower()
        if opt_name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=self.weight_decay,
            )
        if opt_name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=self.weight_decay,
            )
        raise ValueError(f"Unsupported optimizer: {self.optimizer}")

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        epochs=1,
        batch_size=None,
        learning_rate=None,
        **kwargs,
    ):
        X_array = self._ensure_3d_features(X)
        y_array = self._ensure_2d_targets(y)
        if len(X_array) != len(y_array):
            raise ValueError("X and y must contain the same number of windows.")

        epochs = int(self.default_epochs if epochs is None else epochs)
        batch_size = int(self.default_batch_size if batch_size is None else batch_size)
        learning_rate = float(
            self.default_learning_rate if learning_rate is None else learning_rate
        )
        if "weight_decay" in kwargs:
            self.weight_decay = float(kwargs["weight_decay"])

        if self.seed is not None:
            torch.manual_seed(int(self.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.seed))

        self.input_dim = int(X_array.shape[-1])
        self.horizon = int(y_array.shape[-1])
        self.model = self._build_model(self.input_dim, self.horizon)
        self.training_history = {"train_loss": [], "val_loss": []}

        # ITransformerModel uses nn.LazyLinear whose params materialize on first
        # forward. Run a dummy pass before building the optimizer so
        # model.parameters() is complete.
        with torch.no_grad():
            dummy = torch.zeros(1, X_array.shape[1], self.input_dim, device=self.device)
            self.model(dummy)

        train_dataset = TensorDataset(
            torch.from_numpy(X_array),
            torch.from_numpy(y_array),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=max(1, batch_size),
            shuffle=True,
        )

        val_tensors = None
        if X_val is not None and y_val is not None:
            X_val_array = self._ensure_3d_features(X_val)
            y_val_array = self._ensure_2d_targets(y_val)
            val_tensors = (
                torch.from_numpy(X_val_array).to(self.device),
                torch.from_numpy(y_val_array).to(self.device),
            )

        criterion = self._build_loss()
        optimizer = self._build_optimizer(learning_rate)

        for _ in range(max(1, epochs)):
            assert self.model is not None
            self.model.train()
            epoch_losses = []
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))

            self.training_history["train_loss"].append(float(np.mean(epoch_losses)))
            if val_tensors is not None:
                self.model.eval()
                with torch.no_grad():
                    Xv, yv = val_tensors
                    val_loss = criterion(self.model(Xv), yv)
                self.training_history["val_loss"].append(float(val_loss.cpu().item()))

        self.is_fitted = True
        return self

    def predict(self, X):
        if self.model is None or not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        X_array = self._ensure_3d_features(X)
        tensor = torch.from_numpy(X_array).to(self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(tensor).detach().cpu().numpy()
        return np.asarray(predictions, dtype=np.float32).reshape(len(X_array), self.horizon)

    def get_config(self) -> dict:
        return {
            "type": self.model_type,
            "torch_model": self.torch_model_cls.__name__,
            "device": str(self.device),
            "requested_device": self.requested_device,
            "default_epochs": self.default_epochs,
            "default_batch_size": self.default_batch_size,
            "default_learning_rate": self.default_learning_rate,
            "weight_decay": self.weight_decay,
            "loss": self.loss,
            "optimizer": self.optimizer,
            "input_kind": self.input_kind,
            "input_dim": self.input_dim,
            "horizon": self.horizon,
            "is_fitted": self.is_fitted,
            "transformer_config": self._transformer_cfg.as_dict(),
            "history": self.training_history,
        }


# ---------------------------------------------------------------------------
# Experiment 2 concrete adapters
# Defaults match configs/exp2*.yaml
# ---------------------------------------------------------------------------

class DecoderOnlyPipelineModel(_Exp2TorchPipelineModel):
    """Pipeline adapter for Exp.2.a — Decoder-Only Transformer."""

    model_type = "exp2a_decoder_only_transformer"
    torch_model_cls = DecoderOnlyTransformerModel
    _default_transformer_kwargs = {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 3,
        "num_decoder_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "head_hidden_size": 128,
    }


class FFTDecoderPipelineModel(_Exp2TorchPipelineModel):
    """Pipeline adapter for Exp.2.b — FFT + Decoder-Only Transformer."""

    model_type = "exp2b_fft_decoder_transformer"
    torch_model_cls = FFTDecoderTransformerModel
    _default_transformer_kwargs = {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 3,
        "num_decoder_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "head_hidden_size": 128,
        "fft_modes": 32,
    }


class EncoderDecoderPipelineModel(_Exp2TorchPipelineModel):
    """Pipeline adapter for Exp.2.c — Encoder-Decoder Transformer."""

    model_type = "exp2c_encoder_decoder_transformer"
    torch_model_cls = EncoderDecoderTransformerModel
    _default_transformer_kwargs = {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 3,
        "num_decoder_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "head_hidden_size": 128,
    }


class KernelTransformerPipelineModel(_Exp2TorchPipelineModel):
    """Pipeline adapter for Exp.2.d — Kernel (Performer FAVOR+) Transformer."""

    model_type = "exp2d_kernel_transformer"
    torch_model_cls = KernelTransformerModel
    _default_transformer_kwargs = {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "head_hidden_size": 128,
        "feature_dim": 128,
    }


class ITransformerPipelineModel(_Exp2TorchPipelineModel):
    """Pipeline adapter for Exp.2.e — iTransformer (channels-as-tokens)."""

    model_type = "exp2e_itransformer"
    torch_model_cls = ITransformerModel
    _default_transformer_kwargs = {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "head_hidden_size": 128,
    }
