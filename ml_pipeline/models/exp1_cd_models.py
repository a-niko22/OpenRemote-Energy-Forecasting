"""Luis-pipeline adapters for Experiment 1.c and 1.d PyTorch models."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
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
from src.models.cnn_bilstm_transformer import CNNBiLSTMTransformerModel
from src.models.cnn_transformer import CNNTransformerModel


class _TorchForecastingPipelineModel(BaseModel):
    """Small BaseModel adapter around a direct multi-horizon torch module."""

    model_type = "torch_forecasting"
    torch_model_cls: Type[nn.Module]

    def __init__(
        self,
        input_kind: str = "sequence",
        model_config: Any | None = None,
        device: str | torch.device = "auto",
        default_epochs: int = 1,
        default_batch_size: int = 32,
        default_learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        loss: str = "mse",
        optimizer: str = "adam",
        seed: int | None = 42,
    ):
        self.input_kind = input_kind
        self.model_config = model_config
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

    def get_config(self):
        model_config = asdict(self.model_config) if is_dataclass(self.model_config) else self.model_config
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
            "model_config": model_config,
            "history": self.training_history,
        }


class CNNBiLSTMTransformerPipelineModel(_TorchForecastingPipelineModel):
    """Pipeline-compatible wrapper for Exp.1.c."""

    model_type = "exp1c_cnn_bilstm_transformer"
    torch_model_cls = CNNBiLSTMTransformerModel


class CNNTransformerPipelineModel(_TorchForecastingPipelineModel):
    """Pipeline-compatible wrapper for Exp.1.d."""

    model_type = "exp1d_cnn_transformer"
    torch_model_cls = CNNTransformerModel
