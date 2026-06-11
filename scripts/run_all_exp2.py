"""Batch runner for Experiment 2: Transformer variants (2.a – 2.e)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import prepare_run_config, run_experiment, write_batch_outputs


def main() -> None:
    runs = [
        ("configs/exp2a_decoder_only.yaml",      "exp2_standard"),
        ("configs/exp2b_wavelet_decoder.yaml",    "exp2_wavelet"),
        ("configs/exp2c_encoder_decoder.yaml",    "exp2_standard"),
        ("configs/exp2d_kernel.yaml",             "exp2_kernel"),
        ("configs/exp2e_itransformer.yaml",       "exp2_standard"),
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
    write_batch_outputs(
        results,
        output_root=output_root,
        summary_stem="exp2_results",
        summary_title="Experiment 2 Batch Summary",
        summary_description="This summary covers the assigned Experiment 2 Transformer runs.",
    )
    print(results)


if __name__ == "__main__":
    main()
