from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    device: str = 'cpu',
    verbose: bool = True,
) -> Tuple[nn.Module, Dict]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state: dict = {}
    no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_losses.append(criterion(model(X_batch), y_batch).item())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        if verbose and epoch % 10 == 0:
            print(f"    epoch {epoch:4d}/{epochs}  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch}  "
                          f"(best val={best_val_loss:.5f})")
                break

    model.load_state_dict(best_state)
    return model, history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    price_scaler,
    device: str = 'cpu',
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds_list, targets_list = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds_list.append(model(X_batch).cpu().numpy())
            targets_list.append(y_batch.numpy())

    preds   = np.concatenate(preds_list,   axis=0)
    targets = np.concatenate(targets_list, axis=0)

    shape = preds.shape
    preds_orig   = price_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(shape)
    targets_orig = price_scaler.inverse_transform(targets.reshape(-1, 1)).reshape(shape)

    return preds_orig, targets_orig


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Return MAE, RMSE, and MAPE (%) — all in original units."""
    mae  = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))

    # MAPE: skip near-zero targets to avoid inf
    mask = np.abs(targets) > 1e-3
    mape = float(np.mean(np.abs((preds[mask] - targets[mask]) / targets[mask])) * 100)

    return {'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'MAPE': round(mape, 4)}
