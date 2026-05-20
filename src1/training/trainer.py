"""Shared PyTorch trainer for both Experiment 1 models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.training.callbacks import EarlyStoppingState
from src.training.losses import build_loss
from src.utils.config import TrainingConfig


@dataclass
class TrainingResult:
    """Training history and fitted model."""

    model: torch.nn.Module
    history: dict[str, list[float]]
    best_val_loss: float


class Trainer:
    """Train and evaluate a direct multi-step forecasting model."""

    def __init__(self, model: torch.nn.Module, config: TrainingConfig, device: str):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.criterion = build_loss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_patience,
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, logger) -> TrainingResult:
        """Train with early stopping on validation loss."""
        history = {"train_loss": [], "val_loss": []}
        early_stopping = EarlyStoppingState(patience=self.config.patience)

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._run_epoch(train_loader, training=True)
            val_loss = self._run_epoch(val_loader, training=False)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            self.scheduler.step(val_loss)

            logger.info(
                "Epoch %03d/%03d | train_loss=%.6f | val_loss=%.6f | lr=%.6f",
                epoch,
                self.config.epochs,
                train_loss,
                val_loss,
                self.optimizer.param_groups[0]["lr"],
            )

            should_stop = early_stopping.step(self.model, val_loss)
            if should_stop:
                logger.info(
                    "Early stopping triggered at epoch %d with best val_loss %.6f",
                    epoch,
                    early_stopping.best_value,
                )
                break

        if early_stopping.best_state_dict is not None:
            self.model.load_state_dict(early_stopping.best_state_dict)

        return TrainingResult(
            model=self.model,
            history=history,
            best_val_loss=early_stopping.best_value,
        )

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        self.model.train(training)
        losses = []
        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            with torch.set_grad_enabled(training):
                predictions = self.model(inputs)
                loss = self.criterion(predictions, targets)
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip)
                    self.optimizer.step()
            losses.append(loss.detach().item())

        return float(np.mean(losses)) if losses else 0.0

    def predict(self, loader: DataLoader) -> np.ndarray:
        """Predict on one split in scaled space."""
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for inputs, _ in loader:
                batch_predictions = self.model(inputs.to(self.device)).cpu().numpy()
                predictions.append(batch_predictions)
        if not predictions:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(predictions, axis=0)

    def collect_targets(self, loader: DataLoader) -> np.ndarray:
        """Collect targets from one loader."""
        targets = []
        for _, batch_targets in loader:
            targets.append(batch_targets.numpy())
        if not targets:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(targets, axis=0)
