# Experiment 1.a / 1.b Implementation Report

## Scope

This implementation covers only the assigned Experiment 1 variants:

- Exp.1.a: CNN-BiLSTM
- Exp.1.b: CNN-xLSTM

The following variants are intentionally excluded:

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

## Extension Path for Exp.1.c and Exp.1.d

Another teammate can extend this baseline later by:

1. adding new model classes under `src/models/`
2. creating new config files under `configs/`
3. reusing the same data pipeline, preprocessing strategies, trainer, evaluator, and batch summary utilities
4. adding the new runs to a separate batch script or extending the existing runner after team coordination

This keeps the current scope isolated while still making the code easy to extend.
