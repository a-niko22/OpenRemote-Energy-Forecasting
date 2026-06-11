# Experiment 1.a / 1.b Implementation Report

## Scope

This implementation covers only the assigned Experiment 1 variants:

- Exp.1.a: CNN-BiLSTM
- Exp.1.b: CNN-xLSTM

The following variants were intentionally excluded from this historical
Exp.1.a/b results batch:

- Exp.1.c: CNN-BiLSTM-Transformer
- Exp.1.d: CNN-Transformer

This matches the team scope split described in `research/proposal/Sections/ResearchDesign.tex`.

## Implemented Models

### CNN-BiLSTM

- shared temporal CNN encoder
- bidirectional LSTM sequence stage
- direct multi-step regression head

### CNN-xLSTM

- shared temporal CNN encoder
- in-repo xLSTM-inspired recurrent approximation
- direct multi-step regression head

The xLSTM block is explicitly documented as an approximation for reproducibility. It is not presented as an exact implementation of the external official package.

## Preprocessing Methods

All three Experiment 1 preprocessing settings are supported for both implemented models:

1. Normalization
2. Wavelet Transform
3. PatchTST-style patch preprocessing

### Normalization

- standard scaling or min-max scaling
- fit only on the train split
- reused unchanged for validation and test

### Wavelet Transform

- applied per feature on each lookback window
- uses only the current window, so no future information leaks in
- reconstructs approximation and detail signals back to the original lookback length
- supports `concat` and `replace` output modes

### PatchTST-style Patch Preprocessing

- partitions each input window into overlapping temporal patches
- flattens each patch into a token
- lets the model learn a token projection before the CNN stage
- does not introduce a Transformer model into this experiment

## Fairness Considerations

The implementation keeps the comparison fair across all six assigned runs:

- same dataset path by default
- same target column
- same chronological split
- same lookback, horizon, and stride
- same optimizer family
- same early stopping policy
- same metric set
- same random seed policy

## Expected Outputs

Each run writes:

- resolved config
- log file
- scaler artifact
- best model checkpoint
- metrics JSON and CSV
- prediction CSV
- loss plot
- prediction plot
- residual plot
- concise markdown summary

The batch runner also writes:

- `outputs/summary/exp1_ab_results.csv`
- `outputs/summary/exp1_ab_results.json`
- `outputs/summary/exp1_ab_comparison.png`
- `outputs/summary/exp1_ab_summary.md`

## Limitations

- The CNN-xLSTM block is a practical PyTorch approximation, not a direct dependency on the official xLSTM implementation.
- The patch preprocessing is PatchTST-inspired tokenization only.
- The wavelet preprocessing is intentionally lightweight rather than highly specialized.
- No hyperparameter search is included in this baseline.

## Exp.1.c and Exp.1.d Pipeline Integration

Exp.1.c and Exp.1.d now have `ml_pipeline`-compatible wrappers under
`ml_pipeline/models/exp1_cd_models.py`. The original PyTorch modules and
standalone config-based scripts remain available for local validation, while
the shared pipeline can instantiate:

1. `CNNBiLSTMTransformerPipelineModel`
2. `CNNTransformerPipelineModel`

This keeps the historical Exp.1.a/b result scope isolated without implying that
Exp.1.c/d are permanently excluded from the shared pipeline.
