# Exp.1.c/d Local Validation Results Snapshot

## Scope

- Exp.1.c CNN-BiLSTM-Transformer
- Exp.1.d CNN-Transformer

These are local validation runs and do not replace the shared experiment application workflow.

## Preprocessing Methods

- norm
- wavelet
- patch

## Final Test Metrics

| Model | Preprocess | Test MAE | Test RMSE | Test MAPE |
|---|---:|---:|---:|---:|
| exp1c_cnn_bilstm_transformer | norm | 37.6998 | 50.8294 | 4801.9512 |
| exp1c_cnn_bilstm_transformer | patch | 32.2557 | 44.9128 | 4724.6062 |
| exp1c_cnn_bilstm_transformer | wavelet | 39.8661 | 53.1521 | 4896.0546 |
| exp1d_cnn_transformer | norm | 26.5262 | 38.2869 | 3093.2507 |
| exp1d_cnn_transformer | patch | 29.6668 | 40.9629 | 3596.5461 |
| exp1d_cnn_transformer | wavelet | 27.7646 | 39.3107 | 3560.1780 |

## Quick Read

- Best MAE: CNN-Transformer + norm
- Best RMSE: CNN-Transformer + norm
- Best Exp.1.c config: CNN-BiLSTM-Transformer + patch
- Best Exp.1.d config: CNN-Transformer + norm
- Exp.1.c underperformed Exp.1.d across all preprocessing settings

## Interpretation

CNN-Transformer performed better than CNN-BiLSTM-Transformer across the completed Exp.1.c/d runs. Adding BiLSTM before the Transformer did not improve performance in this setup.

CNN-Transformer + norm is the strongest configuration in these local validation runs based on MAE and RMSE.

MAPE should not be used as the main ranking metric because electricity prices near zero make percentage errors unstable. MAE and RMSE are the reliable comparison metrics for these runs.
