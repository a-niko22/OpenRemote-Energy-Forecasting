import os
import sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

ML_PIPELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ML_PIPELINE_ROOT.parent
if str(ML_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import argparse
import random

import numpy as np
import pandas as pd

# VERY IMPORTANT PLS DO NOT TOUCH IMPORTS: import data stack before torch/model adapters,
# otherwise some Windows/server DLL setups can hard-crash with no Python traceback.
from data.loader import chronological_split, load_dataset

from data.windowing import make_windows
from pipeline.experiment import Experiment
from pipeline.experiment_runner import ExperimentRunner, _free_memory
from pipeline.metrics import compute_metrics

from test_models_and_preprocessors.identity_preprocessor import IdentityPreprocessor
from test_models_and_preprocessors.minmax_preprocessor import MinMaxPreprocessor
from test_models_and_preprocessors.mean_model import MeanModel
from test_models_and_preprocessors.zero_model import ZeroModel

from preprocessors.standard_scaler_preprocessor import StandardScalerPreprocessor
from preprocessors.wavelet_preprocessor import WaveletPreprocessor
from preprocessors.patch_tst_preprocessor import PatchTSTPreprocessor


def run_experiments(experiments,
                    repo_id="CitrusBoy/EnergyPriceForecasting",
                    subset="Without_Gas",
                    input_len=168,
                    horizon=48,
                    train_ratio=0.7,
                    val_ratio=0.15,
                    seed=42,
                    **fit_kwargs):
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
    from models.exp1_ab_models import CNNBiLSTMPipelineModel, CNNXLSTMPipelineModel
    from models.exp1_cd_models import CNNBiLSTMTransformerPipelineModel, CNNTransformerPipelineModel

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


def _build_prophet_experiment(args):
    from models.prophet_model import ProphetPipelineModel

    return Experiment(
        "Prophet baseline",
        IdentityPreprocessor(),
        ProphetPipelineModel(
            target_feature_index=args.prophet_target_feature_index,
            freq=args.prophet_freq,
        ),
    )


def _build_exp2_experiments(args):
    from preprocessors.exp2_raw_preprocessors import (
        Exp2StandardPreprocessor,
        Exp2WaveletPreprocessor,
    )
    from preprocessors.kernel_feature_preprocessor import KernelFeaturePreprocessor
    from models.exp2_models import (
        DecoderOnlyTransformerPipelineModel,
        EncoderDecoderTransformerPipelineModel,
        KernelTransformerPipelineModel,
        ITransformerPipelineModel,
    )

    return [
        {
            "kind": "raw",
            "name": "Exp2.a Decoder-Only Tx + Exp2 Standard",
            "preprocessor_factory": lambda: Exp2StandardPreprocessor(),
            "model": _make_model(DecoderOnlyTransformerPipelineModel, seed=args.seed, input_kind="sequence"),
        },
        {
            "kind": "raw",
            "name": "Exp2.b Decoder-Only Tx + Exp2 Wavelet",
            "preprocessor_factory": lambda: Exp2WaveletPreprocessor(),
            "model": _make_model(DecoderOnlyTransformerPipelineModel, seed=args.seed, input_kind="sequence"),
        },
        {
            "kind": "raw",
            "name": "Exp2.c Encoder-Decoder Tx + Exp2 Standard",
            "preprocessor_factory": lambda: Exp2StandardPreprocessor(),
            "model": _make_model(EncoderDecoderTransformerPipelineModel, seed=args.seed, input_kind="sequence"),
        },
        {
            "kind": "windowed",
            "experiment": Experiment(
                "Exp2.d Kernel Tx + Kernel",
                KernelFeaturePreprocessor(
                    n_components=args.kernel_components,
                    gamma=args.kernel_gamma,
                    random_state=args.kernel_random_state,
                ),
                _make_model(KernelTransformerPipelineModel, seed=args.seed, input_kind="sequence"),
            ),
        },
        {
            "kind": "raw",
            "name": "Exp2.e iTransformer + Exp2 Standard",
            "preprocessor_factory": lambda: Exp2StandardPreprocessor(),
            "model": _make_model(ITransformerPipelineModel, seed=args.seed, input_kind="sequence"),
        },
    ]


def _target_values(frame: pd.DataFrame, target_col: str) -> np.ndarray:
    target = pd.to_numeric(frame[target_col], errors="coerce")
    target = target.ffill().bfill().fillna(0.0)
    return target.to_numpy(dtype=np.float32)


def _frame_to_xy(transformed_frame: pd.DataFrame, source_frame: pd.DataFrame, target_col: str):
    X = transformed_frame.to_numpy(dtype=np.float32)
    y = _target_values(source_frame, target_col)
    return X, y


def _release_model(model) -> None:
    try:
        if hasattr(model, "model") and model.model is not None:
            model.model.cpu()
            model.model = None
    except Exception:
        pass
    _free_memory()


def _run_exp2_raw_experiment(run_spec, train_frame, val_frame, test_frame, *,
                             target_col: str, input_len: int, horizon: int,
                             **fit_kwargs):
    preprocessor = run_spec["preprocessor_factory"]()
    model = run_spec["model"]

    preprocessor.fit(train_frame)
    train_transformed = preprocessor.transform(train_frame)
    val_transformed = preprocessor.transform(val_frame)
    test_transformed = preprocessor.transform(test_frame)

    X_train, y_train = _frame_to_xy(train_transformed, train_frame, target_col)
    X_val, y_val = _frame_to_xy(val_transformed, val_frame, target_col)
    X_test, y_test = _frame_to_xy(test_transformed, test_frame, target_col)

    Xw_tr, yw_tr = make_windows(X_train, y_train, input_len, horizon)
    Xw_val, yw_val = make_windows(X_val, y_val, input_len, horizon)
    Xw_te, yw_te = make_windows(X_test, y_test, input_len, horizon)

    model.fit(Xw_tr, yw_tr, X_val=Xw_val, y_val=yw_val, **fit_kwargs)
    predictions = model.predict(Xw_te)

    result = {
        "experiment_name": run_spec["name"],
        "predictions": predictions,
        "y_test": yw_te,
        "input_len": input_len,
        "horizon": horizon,
        "metrics": compute_metrics(yw_te, predictions),
        "model_config": model.get_config(),
        "preprocessor_config": preprocessor.get_config(),
    }
    _release_model(model)
    return result


def run_exp2_experiments(exp2_runs, args, **fit_kwargs):
    from data.loader import chronological_split_frame, load_dataset_frame

    raw_splits = None
    normal_splits = None
    results = []

    for idx, run in enumerate(exp2_runs):
        if run["kind"] == "raw":
            if raw_splits is None:
                exp2_frame = load_dataset_frame(
                    repo_id=args.exp2_repo_id,
                    subset=args.exp2_subset,
                    target_col=args.exp2_target_col,
                )
                raw_splits = chronological_split_frame(
                    exp2_frame,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                )
            print(f"\n[{idx+1}/{len(exp2_runs)}] {run['name']}", flush=True)
            results.append(_run_exp2_raw_experiment(
                run,
                raw_splits[0],
                raw_splits[1],
                raw_splits[2],
                target_col=args.exp2_target_col,
                input_len=args.input_len,
                horizon=args.horizon,
                **fit_kwargs,
            ))
        else:
            if normal_splits is None:
                X, y = load_dataset(subset=args.subset)
                normal_splits = chronological_split(
                    X,
                    y,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                )
            print(f"\n[{idx+1}/{len(exp2_runs)}] {run['experiment'].name}", flush=True)
            runner = ExperimentRunner(input_len=args.input_len, horizon=args.horizon)
            results.append(runner.run(
                run["experiment"],
                normal_splits[0],
                normal_splits[1],
                normal_splits[4],
                normal_splits[5],
                X_val=normal_splits[2],
                y_val=normal_splits[3],
                **fit_kwargs,
            ))
            _release_model(run["experiment"].model)

    return results


def demo():
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

    parser.add_argument("--prophet", action="store_true",
                        help="Run the Prophet baseline first through the shared pipeline.")
    parser.add_argument("--prophet-freq", default="h",
                        help="Pandas frequency string for Prophet's synthetic timeline.")
    parser.add_argument("--prophet-target-feature-index", type=int, default=0,
                        help="Feature index in X that contains the historical target.")

    parser.add_argument("--exp2-repo-id", default="CitrusBoy/EnergyPriceForecasting")
    parser.add_argument("--exp2-subset", default="Gas_Without_Preprocessing")
    parser.add_argument("--exp2-target-col", default="price")

    args = parser.parse_args()

    include_norm = args.include_exp1_norm or args.include_exp1
    include_wavelet = args.include_exp1_wavelet or args.include_exp1
    include_patch = args.include_exp1_patch or args.include_exp1

    experiments = []

    if args.prophet:
        experiments.append(_build_prophet_experiment(args))

    experiments.extend([
        Experiment("Mean baseline", IdentityPreprocessor(), MeanModel()),
        Experiment("Zero baseline", IdentityPreprocessor(), ZeroModel()),
        Experiment("MinMax + Mean", MinMaxPreprocessor(), MeanModel()),
    ])

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

    fit_kwargs = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
    }

    results = run_experiments(
        experiments,
        subset=args.subset,
        input_len=args.input_len,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        **fit_kwargs,
    )

    if args.include_exp2:
        results.extend(run_exp2_experiments(
            _build_exp2_experiments(args),
            args,
            **fit_kwargs,
        ))

    print_results(results, args.input_len, args.horizon)


if __name__ == "__main__":
    demo()