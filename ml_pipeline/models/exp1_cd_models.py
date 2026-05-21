"""Pipeline adapters for Experiment 1.c and 1.d PyTorch models.

This file also defines the shared _TorchForecastingPipelineModel adapter that
all four Exp.1 wrappers inherit from. The adapter has been upgraded to do
realistic training:
  - Early stopping on validation loss
  - Gradient clipping
  - ReduceLROnPlateau scheduler
  - Restores best weights at the end of fit()

These match what src/training/trainer.py does and what the published
reports/exp1_ab_results_snapshot.md numbers were trained with.
"""

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
    """BaseModel adapter around direct multi-horizon torch modules.

    Training loop mirrors src/training/trainer.py: early stopping, grad clip,
    and ReduceLROnPlateau. Defaults match configs/base.yaml so a run without
    explicit kwargs reproduces the conditions used for reports/.
    """

    model_type = "torch_forecasting"
    torch_model_cls: Type[nn.Module]

    def __init__(
        self,
        input_kind: str = "sequence",
        model_config: Any | None = None,
        device: str | torch.device = "auto",
        default_epochs: int = 25,
        default_batch_size: int = 64,
        default_learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        loss: str = "mse",
        optimizer: str = "adamw",
        patience: int = 6,
        grad_clip: float = 1.0,
        lr_scheduler_patience: int = 3,
        lr_scheduler_factor: float = 0.5,
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
        self.patience = int(patience)
        self.grad_clip = float(grad_clip)
        self.lr_scheduler_patience = int(lr_scheduler_patience)
        self.lr_scheduler_factor = float(lr_scheduler_factor)
        self.seed = seed

        self.model: nn.Module | None = None
        self.input_dim: int | None = None
        self.horizon: int | None = None
        self.is_fitted = False
        self.training_history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.best_val_loss: float = float("inf")
        self.stopped_epoch: int | None = None

    # -- setup helpers --------------------------------------------------------

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

    # -- fit / predict --------------------------------------------------------

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        epochs=None,
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
        self.best_val_loss = float("inf")
        self.stopped_epoch = None

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_array), torch.from_numpy(y_array)),
            batch_size=max(1, batch_size),
            shuffle=True,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_array = self._ensure_3d_features(X_val)
            y_val_array = self._ensure_2d_targets(y_val)
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_val_array), torch.from_numpy(y_val_array)),
                batch_size=max(1, batch_size),
                shuffle=False,
            )

        criterion = self._build_loss()
        optimizer = self._build_optimizer(learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
        )

        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0

        for epoch_idx in range(max(1, epochs)):
            # -- train pass --
            assert self.model is not None
            self.model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(xb), yb)
                loss.backward()
                if self.grad_clip is not None and self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))
            train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            self.training_history["train_loss"].append(train_loss)

            # -- val pass + early stopping --
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        val_losses.append(float(criterion(self.model(xb), yb).cpu().item()))
                val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
                self.training_history["val_loss"].append(val_loss)
                scheduler.step(val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in self.model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        self.stopped_epoch = epoch_idx + 1
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_fitted = True
        return self

    def predict(self, X):
        if self.model is None or not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        X_array = self._ensure_3d_features(X)
        if len(X_array) == 0:
            return np.empty((0, self.horizon), dtype=np.float32)

        self.model.eval()
        batch_size = max(1, self.default_batch_size)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X_array), batch_size):
                batch = torch.from_numpy(X_array[start:start + batch_size]).to(self.device)
                outputs.append(self.model(batch).detach().cpu().numpy())
        predictions = np.concatenate(outputs, axis=0)
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
            "patience": self.patience,
            "grad_clip": self.grad_clip,
            "lr_scheduler_patience": self.lr_scheduler_patience,
            "lr_scheduler_factor": self.lr_scheduler_factor,
            "input_kind": self.input_kind,
            "input_dim": self.input_dim,
            "horizon": self.horizon,
            "is_fitted": self.is_fitted,
            "best_val_loss": self.best_val_loss,
            "stopped_epoch": self.stopped_epoch,
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