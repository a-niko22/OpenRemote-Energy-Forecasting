print("[ABSOLUTE TOP OF MAIN]", flush=True)

import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import random
import numpy as np

# VERY IMPORTANT PLS DO NOT TOUCH IMPORTS: import data stack before torch/model adapters, otherwise everything hard crashes and it becomes impossible to debug
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
from preprocessors.kernel_feature_preprocessor import KernelFeaturePreprocessor

# Torch gets imported through these, so keep them AFTER data.loader
from models.exp1_ab_models import CNNBiLSTMPipelineModel, CNNXLSTMPipelineModel
from models.exp1_cd_models import CNNBiLSTMTransformerPipelineModel, CNNTransformerPipelineModel
from models.exp2_models import (
    DecoderOnlyTransformerPipelineModel,
    EncoderDecoderTransformerPipelineModel,
    KernelTransformerPipelineModel,
    ITransformerPipelineModel,
)


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


def _build_exp2_experiments(args):
    """Build the five proposal-aligned Exp.2 experiments."""
    return [
        Experiment(
            "Exp2.a Decoder-Only Tx + Norm",
            StandardScalerPreprocessor(),
            _make_model(DecoderOnlyTransformerPipelineModel, seed=args.seed, input_kind="sequence"),
        ),
        Experiment(
            "Exp2.b Decoder-Only Tx + Wavelet",
            WaveletPreprocessor(
                wavelet_name=args.wavelet_name,
                level=args.wavelet_level,
                mode=args.wavelet_mode,
            ),
            _make_model(DecoderOnlyTransformerPipelineModel, seed=args.seed, input_kind="sequence"),
        ),
        Experiment(
            "Exp2.c Encoder-Decoder Tx + Norm",
            StandardScalerPreprocessor(),
            _make_model(EncoderDecoderTransformerPipelineModel, seed=args.seed, input_kind="sequence"),
        ),
        Experiment(
            "Exp2.d Kernel Tx + Kernel",
            KernelFeaturePreprocessor(
                n_components=args.kernel_components,
                gamma=args.kernel_gamma,
                random_state=args.kernel_random_state,
            ),
            _make_model(KernelTransformerPipelineModel, seed=args.seed, input_kind="sequence"),
        ),
        Experiment(
            "Exp2.e iTransformer + Norm",
            StandardScalerPreprocessor(),
            _make_model(ITransformerPipelineModel, seed=args.seed, input_kind="sequence"),
        ),
    ]


def demo():
    print("[boot] entering demo()", flush=True)

    parser = argparse.ArgumentParser(description="Run Exp.1 and Exp.2 experiments.")
    parser.add_argument("--subset", default="Without_Gas")
    parser.add_argument("--input-len", type=int, default=168)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--include-exp1-norm", action="store_true")
    parser.add_argument("--include-exp1-wavelet", action="store_true")
    parser.add_argument("--include-exp1-patch", action="store_true")
    parser.add_argument("--include-exp1", action="store_true")

    parser.add_argument("--include-exp2", action="store_true",
                        help="Add the 5 proposal-aligned Exp.2 runs.")

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--wavelet-name", default="db1")
    parser.add_argument("--wavelet-level", type=int, default=1)
    parser.add_argument("--wavelet-mode", default="concat", choices=["concat", "replace"])
    parser.add_argument("--patch-len", type=int, default=24)
    parser.add_argument("--patch-stride", type=int, default=12)

    parser.add_argument("--kernel-components", type=int, default=32)
    parser.add_argument("--kernel-gamma", type=float, default=1.0)
    parser.add_argument("--kernel-random-state", type=int, default=42)

    args = parser.parse_args()

    include_norm = args.include_exp1_norm or args.include_exp1
    include_wavelet = args.include_exp1_wavelet or args.include_exp1
    include_patch = args.include_exp1_patch or args.include_exp1

    experiments = [
        Experiment("Mean baseline", IdentityPreprocessor(), MeanModel()),
        Experiment("Zero baseline", IdentityPreprocessor(), ZeroModel()),
        Experiment("MinMax + Mean", MinMaxPreprocessor(), MeanModel()),
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

    if args.include_exp2:
        experiments.extend(_build_exp2_experiments(args))

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
