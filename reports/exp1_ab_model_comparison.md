# Exp.1.a / Exp.1.b Head-to-Head Comparison

This document is the direct comparison view for the two assigned Experiment 1 models:

- `Exp.1.a CNN-BiLSTM`
- `Exp.1.b CNN-xLSTM`

It is meant to make the forecasting differences easier to inspect without jumping between multiple run folders.

## What Changes Between the Models

| Area | `CNN-BiLSTM` | `CNN-xLSTM` |
| --- | --- | --- |
| Shared frontend | Temporal CNN encoder over the input sequence | Same temporal CNN encoder |
| Sequence block | Standard bidirectional LSTM | In-repo xLSTM-inspired recurrent stack |
| Output head | Final recurrent state -> MLP -> direct multi-step forecast | Final recurrent state -> MLP -> direct multi-step forecast |
| Patch support | Accepts patch-tokenized inputs through the same adapter path | Accepts patch-tokenized inputs through the same adapter path |
| Intended behavior | Strong baseline for local + sequential temporal dependencies | Tests whether a more expressive memory block helps on the same forecasting task |
| Main tradeoff observed in practice | Much faster to train | Slower to train, but slightly better test MAE/RMSE in these runs |

## Overall Comparison

The existing aggregate comparison figure is here:

- [Tracked comparison figure](./assets/exp1_ab_comparison.png)

![Overall comparison](./assets/exp1_ab_comparison.png)

## Test Metrics Head to Head

Lower is better for `MAE` and `RMSE`.

| Preprocess | `CNN-BiLSTM` Test MAE | `CNN-xLSTM` Test MAE | MAE Winner | `CNN-BiLSTM` Test RMSE | `CNN-xLSTM` Test RMSE | RMSE Winner |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `norm` | 29.6499 | 26.5345 | `CNN-xLSTM` | 41.2486 | 39.3580 | `CNN-xLSTM` |
| `wavelet` | 28.2966 | 27.1211 | `CNN-xLSTM` | 39.3126 | 39.3053 | `CNN-xLSTM` |
| `patch` | 31.2233 | 28.8341 | `CNN-xLSTM` | 42.6750 | 40.6153 | `CNN-xLSTM` |

## Practical Interpretation

- `CNN-xLSTM` produced the best test metrics in all three preprocessing settings.
- `wavelet` produced the best overall test `RMSE` in the batch: `CNN-xLSTM + wavelet = 39.3053`.
- `norm` produced the best overall test `MAE` in the batch: `CNN-xLSTM + norm = 26.5345`.
- `patch` was the weakest preprocessing strategy for both models on this dataset split.
- `CNN-BiLSTM` remains the more practical baseline when training speed matters.
- `CNN-xLSTM` improved accuracy, but at a much higher training-time cost.

## Training-Time Contrast

Using the completed GPU runs:

| Preprocess | `CNN-BiLSTM` Duration | `CNN-xLSTM` Duration | Slower Model |
| --- | --- | --- | --- |
| `norm` | `00:03:28` | `02:56:06` | `CNN-xLSTM` |
| `wavelet` | `00:03:39` | `03:13:37` | `CNN-xLSTM` |
| `patch` | `00:01:43` | `00:16:14` | `CNN-xLSTM` |

So the head-to-head tradeoff is:

- `CNN-BiLSTM`: faster and simpler
- `CNN-xLSTM`: more expensive, but better test error in this batch

## Forecast Plot Comparison

### `norm`

| `CNN-BiLSTM` | `CNN-xLSTM` |
| --- | --- |
| [Prediction plot](../outputs/exp1a_cnn_bilstm/norm/plots/predictions.png) | [Prediction plot](../outputs/exp1b_cnn_xlstm/norm/plots/predictions.png) |
| [Residual plot](../outputs/exp1a_cnn_bilstm/norm/plots/residuals.png) | [Residual plot](../outputs/exp1b_cnn_xlstm/norm/plots/residuals.png) |
| [Loss curve](../outputs/exp1a_cnn_bilstm/norm/plots/loss.png) | [Loss curve](../outputs/exp1b_cnn_xlstm/norm/plots/loss.png) |

### `wavelet`

| `CNN-BiLSTM` | `CNN-xLSTM` |
| --- | --- |
| [Prediction plot](../outputs/exp1a_cnn_bilstm/wavelet/plots/predictions.png) | [Prediction plot](../outputs/exp1b_cnn_xlstm/wavelet/plots/predictions.png) |
| [Residual plot](../outputs/exp1a_cnn_bilstm/wavelet/plots/residuals.png) | [Residual plot](../outputs/exp1b_cnn_xlstm/wavelet/plots/residuals.png) |
| [Loss curve](../outputs/exp1a_cnn_bilstm/wavelet/plots/loss.png) | [Loss curve](../outputs/exp1b_cnn_xlstm/wavelet/plots/loss.png) |

### `patch`

| `CNN-BiLSTM` | `CNN-xLSTM` |
| --- | --- |
| [Prediction plot](../outputs/exp1a_cnn_bilstm/patch/plots/predictions.png) | [Prediction plot](../outputs/exp1b_cnn_xlstm/patch/plots/predictions.png) |
| [Residual plot](../outputs/exp1a_cnn_bilstm/patch/plots/residuals.png) | [Residual plot](../outputs/exp1b_cnn_xlstm/patch/plots/residuals.png) |
| [Loss curve](../outputs/exp1a_cnn_bilstm/patch/plots/loss.png) | [Loss curve](../outputs/exp1b_cnn_xlstm/patch/plots/loss.png) |

The per-run plots above are generated under `outputs/` when experiments are run locally. They are not committed to git, so the tracked summary files for sharing are:

- `reports/exp1_ab_results_snapshot.md`
- `reports/assets/exp1_ab_comparison.png`

## Recommended Reading Order

1. [Batch summary](../outputs/summary/exp1_ab_summary.md)
2. [This comparison file](./exp1_ab_model_comparison.md)
3. [Execution status](./exp1_ab_execution_status.md)
4. [Implementation report](./exp1_ab_implementation_report.md)

## Important Caveat

`MAPE` is not a stable ranking metric here because the electricity price target includes values near zero. For this comparison, `MAE` and `RMSE` are the better metrics to trust.
