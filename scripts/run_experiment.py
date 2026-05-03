"""CLI entrypoint for one Experiment 1 run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Exp.1.a or Exp.1.b forecasting experiment.")
    parser.add_argument("--config", required=True, help="Path to the YAML experiment config.")
    parser.add_argument("--preprocess", required=True, choices=["norm", "wavelet", "patch"])
    parser.add_argument("--data", default=None, help="Optional dataset path override.")
    parser.add_argument("--device", default=None, help="Optional device override: cpu, cuda, or auto.")
    parser.add_argument("--output-root", default=None, help="Optional output root override.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(
        config_path=args.config,
        preprocess_name=args.preprocess,
        data_path=args.data,
        device=args.device,
        output_root=args.output_root,
        seed=args.seed,
    )
    print(result)


if __name__ == "__main__":
    main()
