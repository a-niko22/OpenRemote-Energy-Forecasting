"""
train.py – Train and save a single energy price forecasting model for production use.

Usage
-----
  python train.py --data dataset_2022_2025.csv --model lstm
  python train.py --data dataset_2022_2025.csv --model transformer --epochs 100
  python train.py --data dataset_2022_2025.csv --model itransformer --d_model 256 --nhead 8

Outputs
-------
  saved_models/{model}.pt            — model weights + metadata
  saved_models/{model}_scalers.pkl   — price & feature scalers (needed for inference)
  predictions/{model}_predictions.csv
  predictions/{model}_metrics.csv
  predictions/{model}_loss.png
  predictions/{model}_predictions.png
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from forecasting.pipeline.data_utils import load_csv, prepare_loaders, save_scalers
from forecasting.models import build_model, ALL_MODELS
from forecasting.models.trainer import train_model, evaluate_model, compute_metrics

def parse_args():
    p = argparse.ArgumentParser(
        description='Train and save an energy price forecasting model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data',        required=True,
                   help='Path to CSV dataset (must contain a "price" column)')
    p.add_argument('--model',       required=True, choices=ALL_MODELS,
                   help='Model architecture to train')

    # Sequence settings
    p.add_argument('--lookback',    type=int,   default=168,
                   help='Input window size (number of time steps)')
    p.add_argument('--horizon',     type=int,   default=48,
                   help='Forecast horizon (number of time steps ahead)')

    # Training settings
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--patience',    type=int,   default=15,
                   help='Early-stopping patience')

    # Model parameters
    p.add_argument('--hidden_size', type=int,   default=128,
                   help='Hidden units for LSTM / GRU / BiLSTM')
    p.add_argument('--num_layers',  type=int,   default=2)
    p.add_argument('--d_model',     type=int,   default=128,
                   help='Embedding dim for Transformer / iTransformer')
    p.add_argument('--nhead',       type=int,   default=8)
    p.add_argument('--dropout',     type=float, default=0.1)

    # Output paths
    p.add_argument('--save_dir',    default='saved_models',
                   help='Directory to save model weights and scalers')
    p.add_argument('--output',      default='predictions',
                   help='Directory to save predictions, metrics, and plots')
    p.add_argument('--device',      default='auto')
    return p.parse_args()

def plot_loss(history: dict, model_name: str, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history['train_loss'], label='Train loss')
    ax.plot(history['val_loss'],   label='Val loss')
    ax.set_title(f'{model_name.upper()} – Training loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE loss (scaled)')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f'{model_name}_loss.png')
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved loss curve -> {path}")


def plot_predictions(preds: np.ndarray, targets: np.ndarray,
                     model_name: str, output_dir: str, n_samples: int = 5):
    """Plot the first n_samples test windows."""
    n = min(n_samples, len(preds))
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(targets[i], label='Actual',    color='black',    linewidth=1.8)
        ax.plot(preds[i],   label='Predicted', color='steelblue', linewidth=1.3)
        ax.set_title(f'Test window {i + 1}')
        ax.set_ylabel('EUR/MWh')
        ax.legend(fontsize=8)
    axes[-1].set_xlabel('Hour ahead')
    plt.suptitle(f'{model_name.upper()} – Test set predictions', y=1.01, fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, f'{model_name}_predictions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved prediction plot -> {path}")

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output,   exist_ok=True)

    device = ('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device == 'auto' else args.device
    print(f"Device: {device}\n")

    print(f"Loading  {args.data} ...")
    df = load_csv(args.data)
    print(f"  Rows: {len(df)}  |  Columns: {list(df.columns)}\n")

    loaders, price_scaler, feat_scaler, n_features = prepare_loaders(
        df, lookback=args.lookback, horizon=args.horizon,
        batch_size=args.batch_size,
    )

    print(f"Building {args.model.upper()} ...")
    model = build_model(
        args.model,
        input_size=n_features,
        seq_len=args.lookback,
        horizon=args.horizon,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        d_model=args.d_model,
        nhead=args.nhead,
        dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}\n")

    print(f"Training (max {args.epochs} epochs, patience={args.patience}) ...")
    model, history = train_model(
        model, loaders['train'], loaders['val'],
        epochs=args.epochs, lr=args.lr,
        patience=args.patience, device=device, verbose=True,
    )

    preds, targets = evaluate_model(model, loaders['test'], price_scaler, device)
    metrics = compute_metrics(preds, targets)

    print(f"\n{'=' * 45}")
    print(f"  Test results  ({args.model.upper()})")
    print(f"{'=' * 45}")
    print(f"  MAE  = {metrics['MAE']:.4f} EUR/MWh")
    print(f"  RMSE = {metrics['RMSE']:.4f} EUR/MWh")
    print(f"  MAPE = {metrics['MAPE']:.2f} %")
    print(f"{'=' * 45}\n")

    model_path  = os.path.join(args.save_dir, f'{args.model}.pt')
    scaler_path = os.path.join(args.save_dir, f'{args.model}_scalers.pkl')

    torch.save({
        'state_dict': model.state_dict(),
        'config': {
            'model':       args.model,
            'input_size':  n_features,
            'seq_len':     args.lookback,
            'horizon':     args.horizon,
            'hidden_size': args.hidden_size,
            'num_layers':  args.num_layers,
            'd_model':     args.d_model,
            'nhead':       args.nhead,
            'dropout':     args.dropout,
        },
        'metrics': metrics,
    }, model_path)
    save_scalers(price_scaler, feat_scaler, scaler_path)

    print(f"Saved model   -> {model_path}")
    print(f"Saved scalers -> {scaler_path}\n")

    N, H = preds.shape
    rows = [
        {'sample': i, 'step': h + 1,
         'actual': round(float(targets[i, h]), 4),
         'predicted': round(float(preds[i, h]), 4),
         'error': round(float(preds[i, h] - targets[i, h]), 4)}
        for i in range(N) for h in range(H)
    ]
    pred_df = pd.DataFrame(rows)
    pred_csv = os.path.join(args.output, f'{args.model}_predictions.csv')
    pred_df.to_csv(pred_csv, index=False)
    print(f"  Saved predictions -> {pred_csv}")

    metrics_csv = os.path.join(args.output, f'{args.model}_metrics.csv')
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)
    print(f"  Saved metrics     -> {metrics_csv}")

    plot_loss(history, args.model, args.output)
    plot_predictions(preds, targets, args.model, args.output)


if __name__ == '__main__':
    main()
