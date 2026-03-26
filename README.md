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

## Usage for Machine Learning

This dataset is intended for training predictive models for energy price forecasting. Potential applications include:

- Time series forecasting of electricity prices.
- Load prediction and price correlation analysis.
- Market analysis and risk assessment.

### Preprocessing Recommendations

- Merge data from different sources based on timestamps.
- Handle missing values and outliers.
- Normalize or scale features as needed for ML algorithms.
- Consider temporal features (e.g., seasonality, holidays).

### Example Workflow

1. Load and explore the CSV files.
2. Preprocess data (cleaning, feature engineering).
3. Train models using libraries like scikit-learn, TensorFlow, or PyTorch.
4. Evaluate model performance on price prediction accuracy.

## License

See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for improvements or additional data sources.