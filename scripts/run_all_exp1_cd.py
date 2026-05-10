"""Optional local validation helper for the six Experiment 1.c/d runs.

This is not the official shared experiment application entrypoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import prepare_run_config, run_experiment, write_batch_outputs


def main() -> None:
    runs = [
        ("configs/exp1c_cnn_bilstm_transformer.yaml", "norm"),
        ("configs/exp1c_cnn_bilstm_transformer.yaml", "wavelet"),
        ("configs/exp1c_cnn_bilstm_transformer.yaml", "patch"),
        ("configs/exp1d_cnn_transformer.yaml", "norm"),
        ("configs/exp1d_cnn_transformer.yaml", "wavelet"),
        ("configs/exp1d_cnn_transformer.yaml", "patch"),
    ]

    results = []
    output_root = None
    for config_path, preprocess_name in runs:
        resolved_config = prepare_run_config(config_path=config_path, preprocess_name=preprocess_name)
        output_root = resolved_config.outputs.root_dir
        result = run_experiment(
            config_path=config_path,
            preprocess_name=preprocess_name,
            seed=resolved_config.seed,
        )
        results.append(result)

    if output_root is None:
        raise RuntimeError("Batch runner did not resolve any output root.")
    write_batch_outputs(results, output_root=output_root, summary_name="exp1_cd")
    print(results)


if __name__ == "__main__":
    main()
