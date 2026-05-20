# Exp.1.a / Exp.1.b Execution Status

## Status

All 6 assigned Experiment 1 runs completed successfully on April 21, 2026.

Included runs:

1. `Exp.1.a CNN-BiLSTM + norm`
2. `Exp.1.a CNN-BiLSTM + wavelet`
3. `Exp.1.a CNN-BiLSTM + patch`
4. `Exp.1.b CNN-xLSTM + norm`
5. `Exp.1.b CNN-xLSTM + wavelet`
6. `Exp.1.b CNN-xLSTM + patch`

Excluded from this historical Exp.1.a/b execution batch:

1. `Exp.1.c CNN-BiLSTM-Transformer`
2. `Exp.1.d CNN-Transformer`

## Runtime Environment

- Workspace: `D:\uni\s7\group\good\OpenRemote-Energy-Forecasting`
- Python runtime: bundled Codex runtime Python
- PyTorch runtime used for final runs: `torch 2.5.1+cu124`
- GPU used: `NVIDIA GeForce RTX 3070 Laptop GPU`
- Device flag used for final runs: `--device cuda`
- Seed: `42`
- Default dataset: `data/processed/final_dataset_full_clean.csv`

## Commands Used

Validation:

```powershell
python -m unittest discover -s tests -v
```

Completed experiment commands:

```powershell
python scripts/run_experiment.py --config configs/exp1a_cnn_bilstm.yaml --preprocess norm --device cuda --output-root outputs
python scripts/run_experiment.py --config configs/exp1a_cnn_bilstm.yaml --preprocess wavelet --device cuda --output-root outputs
python scripts/run_experiment.py --config configs/exp1a_cnn_bilstm.yaml --preprocess patch --device cuda --output-root outputs
python scripts/run_experiment.py --config configs/exp1b_cnn_xlstm.yaml --preprocess norm --device cuda --output-root outputs
python scripts/run_experiment.py --config configs/exp1b_cnn_xlstm.yaml --preprocess wavelet --device cuda --output-root outputs
python scripts/run_experiment.py --config configs/exp1b_cnn_xlstm.yaml --preprocess patch --device cuda --output-root outputs
```

Aggregate summary generation:

```powershell
python -c "from src.experiment import write_batch_outputs; ..."
```

## Final Results

| Experiment | Preprocess | Device | Test MAE | Test RMSE | Test MAPE | Duration |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `exp1a_cnn_bilstm` | `norm` | `cuda` | 29.6499 | 41.2486 | 4114.6316 | 00:03:28 |
| `exp1a_cnn_bilstm` | `wavelet` | `cuda` | 28.2966 | 39.3126 | 3141.0965 | 00:03:39 |
| `exp1a_cnn_bilstm` | `patch` | `cuda` | 31.2233 | 42.6750 | 4408.1151 | 00:01:43 |
| `exp1b_cnn_xlstm` | `norm` | `cuda` | 26.5345 | 39.3580 | 4172.5943 | 02:56:06 |
| `exp1b_cnn_xlstm` | `wavelet` | `cuda` | 27.1211 | 39.3053 | 3918.8283 | 03:13:37 |
| `exp1b_cnn_xlstm` | `patch` | `cuda` | 28.8341 | 40.6153 | 4161.9200 | 00:16:14 |

## Output Locations

Per-run outputs:

- `outputs/exp1a_cnn_bilstm/norm/`
- `outputs/exp1a_cnn_bilstm/wavelet/`
- `outputs/exp1a_cnn_bilstm/patch/`
- `outputs/exp1b_cnn_xlstm/norm/`
- `outputs/exp1b_cnn_xlstm/wavelet/`
- `outputs/exp1b_cnn_xlstm/patch/`

Each run directory contains:

- `resolved_config.yaml`
- `run_summary.json`
- `summary.md`
- `predictions.csv`
- `logs/run.log`
- `metrics/metrics.json`
- `metrics/metrics.csv`
- `plots/loss.png`
- `plots/predictions.png`
- `plots/residuals.png`
- `artifacts/best_model.pt`
- `artifacts/scalers.joblib`

Combined outputs:

- `outputs/summary/exp1_ab_results.csv`
- `outputs/summary/exp1_ab_results.json`
- `outputs/summary/exp1_ab_summary.md`
- `outputs/summary/exp1_ab_comparison.png`

## Notes

- Earlier CPU attempts were intentionally not reused for the final report because they were too slow and were interrupted before full completion.
- One earlier GPU batch timed out while `exp1b_cnn_xlstm + norm` was still writing final artifacts. That partial directory was archived and the run was repeated successfully.
- The `CNN-xLSTM` implementation is the documented in-repo approximation described in the README and implementation report, not the official external xLSTM package.
- The MAPE values are very large because the target series includes values close to zero, which makes percentage errors unstable. MAE and RMSE are the more reliable comparison metrics here.
- Exp.1.c/d are not part of these April 21, 2026 results, but they now have
  shared `ml_pipeline` wrappers in `ml_pipeline/models/exp1_cd_models.py`.

## Recommended Files To Inspect First

1. `outputs/summary/exp1_ab_summary.md`
2. `outputs/summary/exp1_ab_results.csv`
3. `outputs/summary/exp1_ab_comparison.png`
4. `outputs/exp1b_cnn_xlstm/wavelet/summary.md`
5. `outputs/exp1b_cnn_xlstm/wavelet/plots/predictions.png`
