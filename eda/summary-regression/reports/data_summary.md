# Dataset Summary

## Inventory

| dataset | rows | columns | date_start | date_end | frequency | duplicate_rows | duplicate_timestamps | missing_columns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| price_source | 122617 | 2 | 2015-01-04 23:00:00 | 2025-12-31 23:45:00 | mixed (mode 0 days 01:00:00, min 0 days 00:15:00, 2 unique steps) | 0 | 0 | None |
| load_source | 96421 | 2 | 2015-01-01 00:00:00 | 2025-12-31 23:00:00 | mixed (mode 0 days 01:00:00, min 0 days 01:00:00, 2 unique steps) | 0 | 0 | None |
| generation_source | 96421 | 2 | 2015-01-01 00:00:00 | 2025-12-31 23:00:00 | mixed (mode 0 days 01:00:00, min 0 days 01:00:00, 2 unique steps) | 0 | 0 | generation_forecast |
| weather_source | 96432 | 5 | 2015-01-01 00:00:00 | 2025-12-31 23:00:00 | regular (0 days 01:00:00) | 0 | 0 | None |
| final_dataset_full_raw | 96337 | 14 | 2015-01-04 23:00:00 | 2025-12-31 23:00:00 | regular (0 days 01:00:00) | 0 | 0 | total_load, generation_forecast |
| final_dataset_full_clean | 96133 | 14 | 2015-01-04 23:00:00 | 2025-12-31 23:00:00 | mixed (mode 0 days 01:00:00, min 0 days 01:00:00, 4 unique steps) | 0 | 0 | None |
| legacy_weather_variant | 96432 | 5 | 2015-01-01 00:00:00 | 2025-12-31 23:00:00 | regular (0 days 01:00:00) | 0 | 0 | None |
| legacy_combined_weather_price | 96337 | 6 | 2015-01-04 23:00:00 | 2025-12-31 23:00:00 | regular (0 days 01:00:00) | 0 | 0 | None |
| legacy_combined_with_temporal | 96337 | 12 | 2015-01-04 23:00:00 | 2025-12-31 23:00:00 | regular (0 days 01:00:00) | 0 | 0 | None |

## Notes

| item | detail |
| --- | --- |
| Target column | price |
| Primary timestamp column | time |
| Primary exogenous variables | temperature_2m, cloud_cover, wind_speed_10m, shortwave_radiation, total_load, generation_forecast |
| Selected weather source | old/weather_2015_2025_nl_avg_fixed.csv |
| Excluded weather variant | old/weather_2015_2025.csv uses a different timestamp format and different scaling such as cloud_cover 34 instead of 0.236. |
| Price frequency break | First quarter-hour timestamp appears at 2025-01-01 00:15:00 |

## Main Alignment Findings

- Hourly aligned panel rows: 96337
- 2025 quarter-hour target rows: 35040
- First quarter-hour price timestamp: 2025-01-01 00:15:00
- Rows removed by `final_dataset_full_clean.csv` relative to `final_dataset_full_raw.csv`: 204

## Strongest Initial Candidate Features

| feature | missing_pct | std | lag1_autocorr | corr_with_price_t | best_lag_corr | best_abs_corr | best_lag_hours | iqr_outlier_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| total_load | 0.011 | 2477.9593 | 0.9685 | -0.0172 | -0.2001 | 0.2001 | 39 | 1.154 |
| wind_speed_10m | 0.0 | 6.6927 | 0.9818 | -0.1919 | -0.1919 | 0.1919 | 0 | 1.948 |
| shortwave_radiation | 0.0 | 195.9721 | 0.9559 | -0.1032 | -0.1032 | 0.1032 | 0 | 7.611 |
| generation_forecast | 0.212 | 1046.699 | 0.9925 | -0.023 | 0.0821 | 0.0821 | 48 | 4.418 |
| temperature_2m | 0.0 | 6.6122 | 0.994 | -0.0293 | -0.0293 | 0.0293 | 0 | 0.181 |
| cloud_cover | 0.0 | 0.3196 | 0.942 | -0.0007 | -0.018 | 0.018 | 22 | 0.0 |

## Best Price Regression Screens

### Single-variable

| primary_feature | r2 | adj_r2 | explained_variance |
| --- | --- | --- | --- |
| wind_speed_10m | 0.0368 | 0.0368 | 0.0368 |
| shortwave_radiation | 0.0106 | 0.0106 | 0.0106 |
| temperature_2m | 0.0009 | 0.0008 | 0.0009 |
| generation_forecast | 0.0005 | 0.0005 | 0.0005 |
| total_load | 0.0003 | 0.0003 | 0.0003 |
| cloud_cover | 0.0 | -0.0 | 0.0 |

### Two-variable

| primary_feature | r2 | adj_r2 | explained_variance |
| --- | --- | --- | --- |
| wind_speed_10m + generation_forecast | 0.0797 | 0.0796 | 0.0797 |
| wind_speed_10m + shortwave_radiation | 0.0461 | 0.0461 | 0.0461 |
| cloud_cover + wind_speed_10m | 0.0385 | 0.0385 | 0.0385 |
| temperature_2m + wind_speed_10m | 0.0381 | 0.038 | 0.0381 |
| wind_speed_10m + total_load | 0.0369 | 0.0368 | 0.0369 |
| shortwave_radiation + total_load | 0.0123 | 0.0122 | 0.0123 |
| shortwave_radiation + generation_forecast | 0.0123 | 0.0122 | 0.0123 |
| temperature_2m + shortwave_radiation | 0.0116 | 0.0116 | 0.0116 |
