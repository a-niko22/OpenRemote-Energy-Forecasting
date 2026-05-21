print("[ABSOLUTE TOP OF MAIN]", flush=True)

import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import random
import numpy as np

# IMPORTANT: import data stack before torch/model adapters
from data.loader import load_dataset  

from pipeline.experiment import Experiment
from pipeline.experiment_runner import ExperimentRunner

from test_models_and_preprocessors.identity_preprocessor import IdentityPreprocessor
from test_models_and_preprocessors.minmax_preprocessor import MinMaxPreprocessor
from test_models_and_preprocessors.mean_model import MeanModel
from test_models_and_preprocessors.zero_model import ZeroModel

from preprocessors.standard_scaler_preprocessor import StandardScalerPreprocessor
from preprocessors.wavelet_preprocessor import WaveletPreprocessor
from preprocessors.patch_tst_preprocessor import PatchTSTPreprocessor

# Torch gets imported through these, so keep them AFTER data.loader
from models.exp1_ab_models import CNNBiLSTMPipelineModel, CNNXLSTMPipelineModel
from models.exp1_cd_models import CNNBiLSTMTransformerPipelineModel, CNNTransformerPipelineModel


def run_experiments(experiments,
                    repo_id="CitrusBoy/EnergyPriceForecasting",
                    subset="Without_Gas",
                    input_len=168,
                    horizon=48,
                    train_ratio=0.7,
                    val_ratio=0.15,
                    seed=42,
                    **fit_kwargs):
    from data.loader import load_dataset, chronological_split

    random.seed(seed)
    np.random.seed(seed)

    X, y = load_dataset(repo_id, subset)
    X_tr, y_tr, X_val, y_val, X_te, y_te = chronological_split(
        X, y, train_ratio, val_ratio
    )

    runner = ExperimentRunner(input_len=input_len, horizon=horizon)
    return runner.run_all(experiments, X_tr, y_tr, X_te, y_te,
                          X_val=X_val, y_val=y_val, **fit_kwargs)


def print_results(results, input_len, horizon):
    header = (f"{'experiment':40} | {'window':>10} | {'MAE':>8} | "
              f"{'RMSE':>8} | {'peak10-MAE':>11}")
    print(header)
    print("-" * len(header))
    for r in results:
        m = r["metrics"]
        print(f"{r['experiment_name']:40} | "
              f"{r['input_len']:>4}->{r['horizon']:<3} | "
              f"{m['mae']:>8.3f} | "
              f"{m['rmse']:>8.3f} | "
              f"{m['peak_mae_top10']:>11.3f}")


# -- factories for the four real Exp.1 models --------------------------------
# The `input_kind` parameter MUST be "patch" when paired with PatchTSTPreprocessor
# so the model's PatchInputAdapter projects tokens correctly. For norm and
# wavelet preprocessors it stays "sequence".

def _make_model(model_cls, *, seed: int, input_kind: str):
    return model_cls(seed=seed, input_kind=input_kind)


def _build_exp1_block(label: str, preprocessor_factory, *,
                      input_kind: str, seed: int):
    """Build the four Exp.1 experiments for one preprocessing strategy."""
    return [
        Experiment(
            f"Exp1.a CNN-BiLSTM + {label}",
            preprocessor_factory(),
            _make_model(CNNBiLSTMPipelineModel, seed=seed, input_kind=input_kind),
        ),
        Experiment(
            f"Exp1.b CNN-xLSTM + {label}",
            preprocessor_factory(),
            _make_model(CNNXLSTMPipelineModel, seed=seed, input_kind=input_kind),
        ),
        Experiment(
            f"Exp1.c CNN-BiLSTM-Tx + {label}",
            preprocessor_factory(),
            _make_model(CNNBiLSTMTransformerPipelineModel, seed=seed, input_kind=input_kind),
        ),
        Experiment(
            f"Exp1.d CNN-Tx + {label}",
            preprocessor_factory(),
            _make_model(CNNTransformerPipelineModel, seed=seed, input_kind=input_kind),
        ),
    ]


def demo():
    print("[boot] entering demo()", flush=True)
    parser = argparse.ArgumentParser(description="Run Exp.1 experiments.")
    parser.add_argument("--subset", default="Without_Gas")
    parser.add_argument("--input-len", type=int, default=168)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    # Per-preprocessing toggles
    parser.add_argument("--include-exp1-norm", action="store_true",
                        help="Add the 4 Exp.1 runs with StandardScalerPreprocessor.")
    parser.add_argument("--include-exp1-wavelet", action="store_true",
                        help="Add the 4 Exp.1 runs with WaveletPreprocessor.")
    parser.add_argument("--include-exp1-patch", action="store_true",
                        help="Add the 4 Exp.1 runs with PatchTSTPreprocessor.")
    parser.add_argument("--include-exp1", action="store_true",
                        help="Shorthand for --include-exp1-norm "
                             "--include-exp1-wavelet --include-exp1-patch.")

    # Training overrides forwarded to model.fit() via fit_kwargs.
    # Defaults match configs/base.yaml so a plain --include-exp1 run reproduces
    # the training conditions of the published reports.
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Wavelet / patch knobs
    parser.add_argument("--wavelet-name", default="db1")
    parser.add_argument("--wavelet-level", type=int, default=1)
    parser.add_argument("--wavelet-mode", default="concat", choices=["concat", "replace"])
    parser.add_argument("--patch-len", type=int, default=24)
    parser.add_argument("--patch-stride", type=int, default=12)

    args = parser.parse_args()

    include_norm    = args.include_exp1_norm    or args.include_exp1
    include_wavelet = args.include_exp1_wavelet or args.include_exp1
    include_patch   = args.include_exp1_patch   or args.include_exp1

    experiments = [
        Experiment("Mean baseline",  IdentityPreprocessor(), MeanModel()),
        Experiment("Zero baseline",  IdentityPreprocessor(), ZeroModel()),
        Experiment("MinMax + Mean",  MinMaxPreprocessor(),   MeanModel()),
    ]

    if include_norm:
        experiments.extend(_build_exp1_block(
            "Norm",
            preprocessor_factory=lambda: StandardScalerPreprocessor(),
            input_kind="sequence",
            seed=args.seed,
        ))

    if include_wavelet:
        experiments.extend(_build_exp1_block(
            "Wavelet",
            preprocessor_factory=lambda: WaveletPreprocessor(
                wavelet_name=args.wavelet_name,
                level=args.wavelet_level,
                mode=args.wavelet_mode,
            ),
            input_kind="sequence",
            seed=args.seed,
        ))

    if include_patch:
        experiments.extend(_build_exp1_block(
            "Patch",
            preprocessor_factory=lambda: PatchTSTPreprocessor(
                patch_len=args.patch_len,
                patch_stride=args.patch_stride,
            ),
            input_kind="patch",
            seed=args.seed,
        ))

    results = run_experiments(
        experiments,
        subset=args.subset,
        input_len=args.input_len,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    print_results(results, args.input_len, args.horizon)


if __name__ == "__main__":
    demo()