"""
benchmark.py – Compare multiple energy price forecasting models on one dataset.

Usage
-----
  # Benchmark all models
  python benchmark.py --data dataset_2022_2025.csv

  # Benchmark specific models
  python benchmark.py --data dataset_2022_2025.csv --model lstm gru

  # Custom horizon / lookback
  python benchmark.py --data dataset_2022_2025.csv --lookback 336 --horizon 48

  # Save results to a custom directory
  python benchmark.py --data dataset_2022_2025.csv --output my_results/

Outputs (written to --output directory)
----------------------------------------
  metrics.csv                    — MAE / RMSE / MAPE per model
  comparison_mae.png             — bar chart of MAE values
  comparison_rmse.png            — bar chart of RMSE values
  comparison_mape.png            — bar chart of MAPE values
  all_models_comparison.png      — all models on the same test window
  {model}_predictions.png        — individual model prediction plot
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from forecasting.pipeline.data_utils import load_csv, prepare_loaders
from forecasting.models import build_model, ALL_MODELS
from forecasting.models.trainer import train_model, evaluate_model, compute_metrics

def parse_args():
    p = argparse.ArgumentParser(
        description='Benchmark energy price forecasting models.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data',        required=True,
                   help='Path to CSV dataset (must contain a "price" column)')
    p.add_argument('--model',       nargs='+', default=['all'],
                   help=f'Models to benchmark. Use "all" or any subset of: {ALL_MODELS}')

    # Sequence settings
    p.add_argument('--lookback',    type=int, default=168,
                   help='Input window size (number of time steps)')
    p.add_argument('--horizon',     type=int, default=48,
                   help='Forecast horizon (number of time steps ahead)')

    # Training settings
    p.add_argument('--epochs',      type=int,   default=50)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--patience',    type=int,   default=10,
                   help='Early-stopping patience (epochs without val improvement)')

    # Model hyperparameters
    p.add_argument('--hidden_size', type=int,   default=128,
                   help='Hidden units for LSTM / GRU / BiLSTM')
    p.add_argument('--num_layers',  type=int,   default=2)
    p.add_argument('--d_model',     type=int,   default=128,
                   help='Embedding dim for Transformer / iTransformer')
    p.add_argument('--nhead',       type=int,   default=8,
                   help='Number of attention heads')
    p.add_argument('--dropout',     type=float, default=0.1)

    # Output
    p.add_argument('--output',      default='results',
                   help='Directory to save metrics, charts, and prediction plots')
    p.add_argument('--device',      default='auto',
                   help='"cpu", "cuda", or "auto"')
    return p.parse_args()

def plot_predictions(targets: np.ndarray, preds_dict: dict,
                     path: str, title: str = ''):
    """Plot actual vs predicted for one or more models on a single window."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(targets, label='Actual', color='black', linewidth=2)
    colors = plt.cm.tab10.colors
    for i, (name, pred) in enumerate(preds_dict.items()):
        ax.plot(pred, label=name.upper(), color=colors[i % len(colors)], linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel('Hour ahead')
    ax.set_ylabel('Price (EUR/MWh)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison_bars(metrics_df: pd.DataFrame, output_dir: str):
    """One bar chart per metric showing all models side by side."""
    for metric in ['MAE', 'RMSE', 'MAPE']:
        if metric not in metrics_df.columns:
            continue
        vals = metrics_df[metric].sort_values()
        fig, ax = plt.subplots(figsize=(max(6, len(vals) * 1.4), 4))
        bars = ax.bar(vals.index, vals.values, color='steelblue', width=0.5)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
        unit = '%' if metric == 'MAPE' else 'EUR/MWh'
        ax.set_title(f'Model Comparison – {metric}')
        ax.set_ylabel(f'{metric} ({unit})')
        ax.set_xlabel('Model')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{metric.lower()}.png'), dpi=150)
        plt.close(fig)

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = ('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device == 'auto' else args.device
    print(f"Device: {device}\n")

    print(f"Loading  {args.data} ...")
    df = load_csv(args.data)
    print(f"  Rows: {len(df)}  |  Columns: {list(df.columns)}\n")

    loaders, price_scaler, _, n_features = prepare_loaders(
        df, lookback=args.lookback, horizon=args.horizon,
        batch_size=args.batch_size,
    )
    print(f"  Input features : {n_features}")
    print(f"  Lookback       : {args.lookback} steps")
    print(f"  Horizon        : {args.horizon} steps\n")

    if 'all' in args.model:
        models_to_run = ALL_MODELS
    else:
        models_to_run = [m.lower() for m in args.model]

    results: dict = {}
    all_preds: dict = {}
    all_targets = None

    for name in models_to_run:
        print(f"{'-' * 55}")
        print(f"  {name.upper()}")
        print(f"{'-' * 55}")
        t0 = time.time()

        model = build_model(
            name,
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
        print(f"  Parameters: {n_params:,}")

        model, _ = train_model(
            model, loaders['train'], loaders['val'],
            epochs=args.epochs, lr=args.lr,
            patience=args.patience, device=device, verbose=True,
        )

        preds, targets = evaluate_model(model, loaders['test'], price_scaler, device)
        metrics = compute_metrics(preds, targets)
        elapsed = time.time() - t0

        print(f"\n  >> MAE={metrics['MAE']:.3f}  "
              f"RMSE={metrics['RMSE']:.3f}  "
              f"MAPE={metrics['MAPE']:.2f}%  "
              f"time={elapsed:.1f}s\n")

        results[name] = {**metrics, 'params': n_params, 'time_s': round(elapsed, 1)}
        all_preds[name] = preds
        all_targets = targets

        plot_predictions(
            targets[0],
            {name: preds[0]},
            os.path.join(args.output, f'{name}_predictions.png'),
            title=f'{name.upper()} – First Test Window (48 h)',
        )

    metrics_df = pd.DataFrame(results).T.round(4)
    metrics_df.index.name = 'model'

    sep = '=' * 60
    print(f"\n{sep}")
    print("  BENCHMARK RESULTS")
    print(sep)
    print(metrics_df[['MAE', 'RMSE', 'MAPE', 'time_s']].to_string())
    print(sep)

    best_mae  = metrics_df['MAE'].idxmin()
    best_rmse = metrics_df['RMSE'].idxmin()
    print(f"\n  Best MAE  -> {best_mae.upper()}: {metrics_df.loc[best_mae,  'MAE']:.4f} EUR/MWh")
    print(f"  Best RMSE -> {best_rmse.upper()}: {metrics_df.loc[best_rmse, 'RMSE']:.4f} EUR/MWh")

    csv_path = os.path.join(args.output, 'metrics.csv')
    metrics_df.to_csv(csv_path)
    print(f"\nSaved -> {csv_path}")

    plot_comparison_bars(metrics_df, args.output)
    print(f"Saved -> {args.output}/comparison_*.png")

    if len(models_to_run) > 1 and all_targets is not None:
        plot_predictions(
            all_targets[0],
            {n: all_preds[n][0] for n in models_to_run},
            os.path.join(args.output, 'all_models_comparison.png'),
            title='All Models – First Test Window (48 h)',
        )
        print(f"Saved -> {args.output}/all_models_comparison.png")


if __name__ == '__main__':
    main()
