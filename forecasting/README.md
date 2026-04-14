# Energy Forecasting Dataset

This repository contains a curated dataset for training machine learning models to predict energy prices in the wholesale electricity market in the Netherlands. The dataset aggregates historical data from multiple European sources, focusing on electricity prices, load forecasts, and settlement prices.

## Data Sources

The dataset is organized into three main sources:

### EMBER (European Market for Electricity)
- **File**: `data/ember/european_wholesale_electricity_price_data_monthly.csv`
- **Description**: Monthly wholesale electricity price data across European markets.
- **Coverage**: Historical monthly data.

### ENTSO-E Transparency Platform
- **Load Data**: Located in `data/entso-transparency-platform/load/`
  - Files: `GUI_TOTAL_LOAD_DAYAHEAD_[year-range].csv` (e.g., 2014-2015, 2015-2016, ..., 2024-2025)
  - **Description**: Day-ahead total load forecasts for the GUI region (likely Germany/France/Italy).
  - **Coverage**: Daily data from 2014 to 2025.

- **Price Data**: Located in `data/entso-transparency-platform/prices/`
  - Files: `GUI_ENERGY_PRICES_[year-range].csv` (e.g., 2015-2016, 2016-2017, ..., 2025-2026)
  - **Description**: Energy price data for the GUI region.
  - **Coverage**: Hourly/daily prices from 2015 to 2026.

### Tennet (Netherlands Transmission System Operator)
- **Files**: Located in `data/tennet/`
  - `settlement_prices_[year-range].csv` (e.g., 2018-2019, 2019-2020, ..., 2024-2025)
- **Description**: Settlement prices for electricity in the Netherlands.
- **Coverage**: Historical settlement prices from 2018 to 2025.

## Dataset Structure

- All data is in CSV format.
- Columns vary by source but typically include timestamps, prices, loads, and regional identifiers.
- Time ranges: From 2014 to 2026, with varying granularity (monthly, daily, hourly).

## Forecasting Pipeline

A full PyTorch-based forecasting pipeline is included in this repository.

### Installation

```bash
pip install -r requirements.txt
```

> **GPU (CUDA) users:** The pipeline auto-detects CUDA. For RTX 5000-series (Blackwell) GPUs, install the nightly PyTorch build:
> ```bash
> pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
> ```

### Compiled Dataset

Raw data from the sources above is merged into a single ready-to-use CSV:

- **`dataset_2022_2025.csv`** — hourly data (2022–2025) with columns:
  `price`, `gen_day_ahead`, `gen_intraday`, `gen_actual`, `load_actual`,
  `load_forecast`, `temp_2m`, `cloud_cover`, `wind_speed_10m`,
  `shortwave_radiation`, `hour`, `day_of_week`, `month`

### Model Architectures

Five architectures are available (`forecasting/models.py`):

| Model | Description |
|---|---|
| `lstm` | Stacked LSTM encoder |
| `gru` | Stacked GRU encoder |
| `bilstm` | Bidirectional LSTM |
| `transformer` | Vanilla Transformer encoder (time steps as tokens) |
| `itransformer` | Inverted Transformer — variates as tokens (Liu et al., ICLR 2024) |

### Training a Single Model

```bash
python train.py --data dataset_2022_2025.csv --model lstm
python train.py --data dataset_2022_2025.csv --model transformer --epochs 100
python train.py --data dataset_2022_2025.csv --model itransformer --d_model 256 --nhead 8
```

Outputs saved to `saved_models/` (weights + scalers) and `predictions/` (CSV, plots).

### Benchmarking All Models

```bash
python benchmark.py --data dataset_2022_2025.csv
python benchmark.py --data dataset_2022_2025.csv --model lstm gru --epochs 50
```

Outputs saved to `results/`: per-model metrics (MAE, RMSE, MAPE), bar charts, and prediction plots.

### Key CLI Options

| Flag | Default | Description |
|---|---|---|
| `--lookback` | 168 | Input window size (hours) |
| `--horizon` | 48 | Forecast horizon (hours ahead) |
| `--epochs` | 100 / 50 | Max training epochs |
| `--batch_size` | 32 | Batch size |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |

### Preprocessing

The pipeline (`forecasting/data_utils.py`) handles:
- Automatic timestamp detection
- Dropping columns with >50% missing values
- Forward/back-fill for remaining NaNs
- Separate `StandardScaler` for price (target) and features
- Sliding-window dataset with 75/10/15 train/val/test split

## License

See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for improvements or additional data sources.