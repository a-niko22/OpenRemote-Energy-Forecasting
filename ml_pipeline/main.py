# main.py
# =============================================================================
# Entry point for the  pipeline.
#
# This file is the only place you need to touch to run experiments. The
# pipeline itself (data loading, splitting, windowing, fit/transform, metrics)
# is already wired up. Your job is to define WHICH experiments to run by
# adding entries to the `experiments` list inside demo().
#
# Pipeline flow (handled for you, do not modify):
#   load_dataset -> chronological_split -> windowing -> preprocessor.fit on
#   train -> preprocessor.transform on train/val/test -> model.fit -> predict
#   -> metrics.
#
# What you need to do:
#   1. Build your model in a new file under models/ that inherits from
#      BaseModel (interfaces/base_model.py).
#   2. Build your preprocessor under preprocessors/ that inherits
#      from BasePreprocessor (interfaces/base_preprocessor.py).
#      If you don't need preprocessing, use IdentityPreprocessor inside test_models_and_preprocessors/.
#   3. Import them at the top of this file.
#   4. Add an Experiment(...) entry to the `experiments` list in demo().
#   5. Run: python main.py
#
# Contracts (read these before implementing):
#   BaseModel        -> fit(X, y, X_val=None, y_val=None, **kwargs), predict(X)
#                       X shape: (n_windows, input_len, n_features)
#                       y shape: (n_windows, horizon)
#                       predict returns (n_windows, horizon)
#   BasePreprocessor -> fit(X, y=None), transform(X)
#                       MUST be shape-preserving. Do not change row count.
#                       Windowing is NOT a preprocessor (it changes shape).
# =============================================================================

import argparse
import random
import numpy as np

from data.loader import load_dataset, chronological_split
from pipeline.experiment import Experiment
from pipeline.experiment_runner import ExperimentRunner

# Baselines provided as reference implementations. Use these to sanity-check
# that any real model you build actually learns something. If your model
# cannot beat MeanModel, it is broken or sucks.
from test_models_and_preprocessors.identity_preprocessor import IdentityPreprocessor
from test_models_and_preprocessors.minmax_preprocessor import MinMaxPreprocessor
from test_models_and_preprocessors.mean_model import MeanModel
from test_models_and_preprocessors.zero_model import ZeroModel

# import your own models and preprocessors here



def run_experiments(experiments,
                    repo_id="CitrusBoy/EnergyPriceForecasting",
                    subset="Without_Gas",
                    input_len=168,
                    horizon=48,
                    train_ratio=0.7,
                    val_ratio=0.15,
                    seed=42):
    """Core application entrypoint. Takes a list of experiments, runs them all.

    Experiments can be built elsewhere and handed in. The app handles data
    loading, splitting, windowing, and result collection, not which
    experiments to run.

    You normally do not need to call this directly. demo() wraps it with CLI
    arguments. Call this from a notebook if you want to script runs.
    """
    # Seed everything we control here. If your model uses PyTorch/TF, ALSO seed
    # those inside your model's __init__ or fit() (torch.manual_seed,
    # tf.random.set_seed). The pipeline cannot do that for you.
    random.seed(seed)
    np.random.seed(seed)

    # Loads from Hugging Face. First run downloads, subsequent runs hit the
    # local cache (~/.cache/huggingface/datasets). Subset options live in the
    # dataset repo: "Gas", "Without_Gas"
    X, y = load_dataset(repo_id, subset)

    # Chronological split: NO shuffling. Time-series data must stay ordered or
    # we leak future info into training. Default 70/15/15.
    X_tr, y_tr, X_val, y_val, X_te, y_te = chronological_split(
        X, y, train_ratio, val_ratio
    )

    # Runner does per-experiment windowing with the input_len/horizon below,
    # unless an Experiment overrides them via its own input_len/horizon args.
    runner = ExperimentRunner(input_len=input_len, horizon=horizon)
    return runner.run_all(experiments, X_tr, y_tr, X_te, y_te,
                          X_val=X_val, y_val=y_val)


def print_results(results, input_len, horizon):
    """Pretty-prints the metrics table. Add columns here if you add metrics
    in pipeline/metrics.py.

    Current metrics:
      MAE         -> overall mean absolute error
      RMSE        -> overall root mean squared error
      peak10-MAE  -> MAE on the top 10% of true prices (beat Prophet on peaks specifically)
    Per-horizon MAE is computed but not printed; pull it from
    results[i]["metrics"]["per_horizon_mae"] if you want to plot error growth
    across the 48h horizon
    """
    header = f"{'experiment':20} | {'window':>10} | {'MAE':>8} | {'RMSE':>8} | {'peak10-MAE':>11}"
    print(header)
    print("-" * len(header))
    for r in results:
        m = r["metrics"]
        print(f"{r['experiment_name']:20} | "
              f"{r['input_len']:>4}->{r['horizon']:<3} | "
              f"{m['mae']:>8.3f} | "
              f"{m['rmse']:>8.3f} | "
              f"{m['peak_mae_top10']:>11.3f}")


def demo():
    """Runs the configured experiments. THIS is the function you edit.

    CLI usage:
      python main.py                          # run with defaults
      python main.py --subset Gas             # switch dataset subset
      python main.py --horizon 24             # 24h forecast instead of 48h
      python main.py --input-len 336          # 2-week lookback instead of 1
      python main.py --seed 0                 # different seed for reproducibility checks
    """
    parser = argparse.ArgumentParser(description="Run baseline experiments.")
    parser.add_argument("--subset", default="Without_Gas")
    parser.add_argument("--input-len", type=int, default=168,
                        help="Lookback window size in hours.")
    parser.add_argument("--horizon", type=int, default=48,
                        help="Forecast horizon in hours.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # =========================================================================
    # EDIT HERE: Define experiments to run.
    #
    # Each entry is Experiment(name, preprocessor, model, input_len=?, horizon=?).
    # input_len/horizon are optional per-experiment overrides; if omitted the
    # CLI defaults are used.
    #
    # Rules:
    #   - name must be unique (it shows up in the results table).
    #   - preprocessor must inherit from BasePreprocessor.
    #   - model must inherit from BaseModel.
    #   - Use IdentityPreprocessor when your model needs raw input.
    #
    # The three baselines below can stay in the list. Every real model you
    # build can be compared against these. If a real model loses to
    # MeanModel, it is broken or really bad; do not report it as a result.

    # =========================================================================
    experiments = [
        Experiment("Mean baseline",  IdentityPreprocessor(), MeanModel()),
        Experiment("Zero baseline",  IdentityPreprocessor(), ZeroModel()),
        Experiment("MinMax + Mean",  MinMaxPreprocessor(),   MeanModel()),
        # TODO: add your experiments below this line.
    ]

    results = run_experiments(
        experiments,
        subset=args.subset,
        input_len=args.input_len,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print_results(results, args.input_len, args.horizon)


if __name__ == "__main__":
    demo()