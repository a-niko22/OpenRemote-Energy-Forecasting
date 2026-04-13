# EDA Report: EPEX Electricity Price Forecasting

## 1. Introduction

This report documents a reproducible exploratory data analysis for the current electricity price forecasting dataset. The project goal is to forecast EPEX electricity prices 48 hours ahead at 15-minute intervals, but the available exogenous sources are still hourly. The analysis therefore splits into two tracks:

- an **hourly aligned panel** for joint price, production, and weather analysis
- a **2025 15-minute price-only segment** to document the current resolution change in the target series

All timestamps were parsed as naive datetimes. This was deliberate because the repository documentation says UTC, while the load and generation files show daylight-saving spring-forward gaps that are inconsistent with a clean UTC timeline.

## 2. Dataset Overview

### Source inventory

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

### Key inventory notes

| item | detail |
| --- | --- |
| Target column | price |
| Primary timestamp column | time |
| Primary exogenous variables | temperature_2m, cloud_cover, wind_speed_10m, shortwave_radiation, total_load, generation_forecast |
| Selected weather source | old/weather_2015_2025_nl_avg_fixed.csv |
| Excluded weather variant | old/weather_2015_2025.csv uses a different timestamp format and different scaling such as cloud_cover 34 instead of 0.236. |
| Price frequency break | First quarter-hour timestamp appears at 2025-01-01 00:15:00 |

### Combined-panel audit

| dataset | rows | missing_times_vs_hourly_panel | extra_times_vs_hourly_panel |
| --- | --- | --- | --- |
| hourly_panel_raw | 96337 | 0 | 0 |
| final_dataset_full_raw | 96337 | 0 | 0 |
| final_dataset_full_clean | 96133 | 204 | 0 |

The source-based hourly panel keeps nulls for auditability. The prebuilt clean dataset removes rows entirely when exogenous values are absent, which shortens the usable sample and hides the original missing-value pattern.

## 3. Data Alignment and Time Handling

The source data does not align cleanly to a single operational forecasting panel:

- The price series is hourly-only through **2025-01-01 00:00:00** and then expands to quarter-hour resolution from **2025-01-01 00:15:00** onward.
- Weather, load, and generation remain hourly across the full period.
- Load and generation show spring-forward gaps at `02:00` on daylight-saving transition dates, while price and weather still contain those timestamps.
- The hourly aligned panel was constructed by keeping only top-of-hour price observations and left-joining weather, load, and generation on exact timestamps.

![Source coverage timeline](figures/static/source_coverage_timeline.png)

![Price resolution transition](figures/static/price_frequency_transition.png)

![Detected source gaps](figures/static/source_gap_events.png)

## 4. Data Quality Findings

### Missingness and stability

| column | missing_count | missing_pct | negative_count | zero_count | iqr_outlier_count | robust_anomaly_count | std | unique_values |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generation_forecast | 204 | 0.212 | 0 | 0 | 4256 | 2600 | 1046.699 | 22263 |
| total_load | 11 | 0.011 | 0 | 0 | 1112 | 58 | 2477.9593 | 94871 |
| shortwave_radiation | 0 | 0.0 | 0 | 44457 | 7332 | 41483 | 195.9721 | 4168 |
| price | 0 | 0.0 | 1608 | 259 | 8088 | 8348 | 78.7906 | 21148 |
| wind_speed_10m | 0 | 0.0 | 0 | 0 | 1877 | 271 | 6.6927 | 2461 |
| temperature_2m | 0 | 0.0 | 3525 | 11 | 174 | 0 | 6.6122 | 2521 |
| cloud_cover | 0 | 0.0 | 0 | 2510 | 0 | 0 | 0.3196 | 501 |

### Rule-based suspicious values

| column | rule | suspicious_count |
| --- | --- | --- |
| temperature_2m | [-40.0, 50.0] | 0 |
| cloud_cover | [0.0, 1.0] | 0 |
| wind_speed_10m | [0.0, 60.0] | 0 |
| shortwave_radiation | [0.0, 1400.0] | 0 |
| total_load | [0.0, None] | 0 |
| generation_forecast | [0.0, None] | 0 |
| price | [-500.0, 1000.0] | 0 |

### Missing timestamps removed by the prebuilt clean dataset

| time_removed_in_clean |
| --- |
| 2015-02-07 01:00:00 |
| 2015-02-07 02:00:00 |
| 2015-02-07 03:00:00 |
| 2015-02-07 04:00:00 |
| 2015-02-07 05:00:00 |
| 2015-02-07 06:00:00 |
| 2015-02-07 07:00:00 |
| 2015-02-07 08:00:00 |
| 2015-02-07 09:00:00 |
| 2015-02-07 10:00:00 |

The main quality issues are concentrated in the exogenous system variables rather than the target itself. Missingness is dominated by blocks in `generation_forecast`, plus repeated DST-related nulls in `total_load` and `generation_forecast` at the missing `02:00` hour in spring. These rows should be reported and handled explicitly during modeling rather than silently dropped up front.

![Missingness by column](figures/static/missingness_overview.png)

![Missingness heatmap](figures/static/missingness_heatmap.png)

![Missingness over time](figures/static/missingness_over_time.png)

## 5. Target Series Analysis

The hourly target series shows clear intraday structure, changing volatility, and episodic spikes. Negative prices are present, so models must preserve them rather than clipping them away. The distribution is skewed and heavy-tailed, and the volatility profile changes materially over time, which argues against assuming a stationary low-variance process.

- Negative hourly prices observed: **1608**
- Hourly price mean: **76.62 EUR/MWh**
- Hourly price standard deviation: **78.79 EUR/MWh**
- Hourly price min / max: **-500.00 / 872.96 EUR/MWh**

![Full price series](figures/static/price_full_series.png)

![Representative month](figures/static/price_month_zoom.png)

![Representative week](figures/static/price_week_zoom.png)

![Representative day](figures/static/price_day_zoom.png)

![Price distribution](figures/static/price_distribution.png)

![Price by hour](figures/static/price_boxplot_by_hour.png)

![Price by day of week](figures/static/price_boxplot_by_day_of_week.png)

![Rolling mean](figures/static/price_rolling_mean.png)

![Rolling volatility](figures/static/price_rolling_std.png)

![Weekday vs weekend](figures/static/price_weekday_vs_weekend.png)

![Price ACF](figures/static/price_acf.png)

![Price PACF](figures/static/price_pacf.png)

![Seasonal decomposition](figures/static/price_seasonal_decomposition.png)

### 2025 15-minute price-only segment

The 2025 target series provides the quarter-hour resolution required by the project goal, but only for the target. Because the exogenous sources remain hourly, this section is descriptive only and should not be mistaken for a fully aligned 15-minute forecasting panel.

![2025 15-minute full series](figures/static/price_2025_15m_full_series.png)

![2025 month zoom](figures/static/price_2025_15m_month_zoom.png)

![2025 week zoom](figures/static/price_2025_15m_week_zoom.png)

![2025 day zoom](figures/static/price_2025_15m_day_zoom.png)

![2025 distribution](figures/static/price_2025_15m_distribution.png)

## 6. Weather and Production Feature Analysis

The exogenous variables are informative but not equally strong. The system-level variables tend to have a clearer relationship with price than the raw weather variables, while weather remains useful as a causal driver for renewable output and demand conditions.

### Initial feature screening

| feature | corr_with_price_t | best_abs_corr | best_lag_hours |
| --- | --- | --- | --- |
| total_load | -0.0172 | 0.2001 | 39 |
| wind_speed_10m | -0.1919 | 0.1919 | 0 |
| shortwave_radiation | -0.1032 | 0.1032 | 0 |
| generation_forecast | -0.023 | 0.0821 | 48 |
| temperature_2m | -0.0293 | 0.0293 | 0 |
| cloud_cover | -0.0007 | 0.018 | 22 |

### Signal-quality screen

| feature | missing_pct | std | lag1_autocorr | corr_with_price_t | best_lag_corr | best_abs_corr | best_lag_hours | iqr_outlier_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| total_load | 0.011 | 2477.9593 | 0.9685 | -0.0172 | -0.2001 | 0.2001 | 39 | 1.154 |
| wind_speed_10m | 0.0 | 6.6927 | 0.9818 | -0.1919 | -0.1919 | 0.1919 | 0 | 1.948 |
| shortwave_radiation | 0.0 | 195.9721 | 0.9559 | -0.1032 | -0.1032 | 0.1032 | 0 | 7.611 |
| generation_forecast | 0.212 | 1046.699 | 0.9925 | -0.023 | 0.0821 | 0.0821 | 48 | 4.418 |
| temperature_2m | 0.0 | 6.6122 | 0.994 | -0.0293 | -0.0293 | 0.0293 | 0 | 0.181 |
| cloud_cover | 0.0 | 0.3196 | 0.942 | -0.0007 | -0.018 | 0.018 | 22 | 0.0 |

![Weather over time](figures/static/weather_timeseries.png)

![Production over time](figures/static/production_timeseries.png)

![Weather distributions](figures/static/weather_feature_histograms.png)

![Production distributions](figures/static/production_feature_histograms.png)

![Correlation heatmap](figures/static/correlation_heatmap_weather_production_price.png)

![Scatter screening](figures/static/price_vs_exogenous_scatter_grid.png)

![Lagged correlations](figures/static/lagged_correlation_screen.png)

The lagged-correlation screen is only an exploratory ranking tool. It helps identify plausible variables and lags for later forecasting features, but it does not prove predictive value at a 48-hour horizon.

### Weather vs Price and Load R² Screening

The teammate request for weather-to-price and weather-to-load `R²` is implemented here as **pairwise simple OLS**. Each model uses one weather variable at a time and a single target. In this context:

- **R²** measures how much same-time variance in the target is explained by one linear weather predictor.
- **Model p-value** tests whether the overall fitted linear relationship is distinguishable from zero under the OLS assumptions.
- **Explained variance** is included alongside `R²` because it gives another summary of how much fitted variation the model captures, using the model predictions directly.

These statistics are useful for screening and practice, but they are **not forecasting metrics**. A low pairwise `R²` does not mean the feature is useless later, because lagged effects, interactions, nonlinear models, and joint modeling can still make the feature valuable.

#### Pairwise weather OLS ranking

| feature | r2_price | r2_total_load | delta_r2_load_minus_price | recommendation |
| --- | --- | --- | --- | --- |
| temperature_2m | 0.0008584139496191634 | 0.06311409467948348 | 0.062255680729864316 | better for load |
| shortwave_radiation | 0.010649294737977932 | 0.04472858082477693 | 0.034079286086799 | better for load |
| cloud_cover | 5.112628925774132e-07 | 0.005625414785741634 | 0.005624903522849056 | weak direct linear signal |
| wind_speed_10m | 0.03684338161958123 | 0.0037058301453067877 | -0.03313755147427444 | better for direct price screening |

#### Best weather predictors for price

| feature | r2 | adj_r2 | explained_variance | model_pvalue | nobs |
| --- | --- | --- | --- | --- | --- |
| wind_speed_10m | 0.03684338161958123 | 0.036833383626968086 | 0.036843381619580895 | 0.0 | 96337 |
| shortwave_radiation | 0.010649294737977932 | 0.010639024839132594 | 0.010649294737978154 | 2.6914597519202063e-226 | 96337 |
| temperature_2m | 0.0008584139496191634 | 0.0008480424170914658 | 0.0008584139496191634 | 9.40440893167978e-20 | 96337 |
| cloud_cover | 5.112628925774132e-07 | -9.869175045196243e-06 | 5.112628926884355e-07 | 0.8243695288125994 | 96337 |

#### Best weather predictors for total load

| feature | r2 | adj_r2 | explained_variance | model_pvalue | nobs |
| --- | --- | --- | --- | --- | --- |
| temperature_2m | 0.06311409467948348 | 0.06310436827790844 | 0.0631140946794837 | 0.0 | 96326 |
| shortwave_radiation | 0.04472858082477693 | 0.04471866355162413 | 0.0447285808247766 | 0.0 | 96326 |
| cloud_cover | 0.005625414785741634 | 0.005615091558039187 | 0.005625414785741745 | 3.451942800944583e-120 | 96326 |
| wind_speed_10m | 0.0037058301453067877 | 0.0036954869891894226 | 0.0037058301453066766 | 9.27859284840457e-80 | 96326 |

![Weather R² vs price](figures/static/weather_r2_vs_price.png)

![Weather R² vs total load](figures/static/weather_r2_vs_load.png)

![Weather R² comparison](figures/static/weather_r2_price_load_comparison.png)

## 7. Exploratory Regression Analysis

Ordinary least squares was used here as an interpretation tool, not as the final forecasting benchmark. The models were fit on the hourly aligned panel with per-model row dropping where needed.

### Regression metrics

| model_name | target_column | model_type | primary_feature | feature_count | feature_list | nobs | dropped_rows | r2 | adj_r2 | explained_variance | model_pvalue | aic | bic | summary_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Multiple OLS B: price ~ weather + production + calendar | price | multiple | temperature_2m, cloud_cover, wind_speed_10m, shortwave_radiation, total_load, generation_forecast, hour_sin, hour_cos, day_of_week, month, is_weekend | 11 | temperature_2m, cloud_cover, wind_speed_10m, shortwave_radiation, total_load, generation_forecast, hour_sin, hour_cos, day_of_week, month, is_weekend | 96133 | 204 | 0.1502 | 0.1501 | 0.1502 | 0.0 | 1096938.44 | 1097052.12 | regression_outputs/multiple_model_b_summary.txt |
| Multiple OLS A: price ~ weather + production | price | multiple | temperature_2m, cloud_cover, wind_speed_10m, shortwave_radiation, total_load, generation_forecast | 6 | temperature_2m, cloud_cover, wind_speed_10m, shortwave_radiation, total_load, generation_forecast | 96133 | 204 | 0.0824 | 0.0823 | 0.0824 | 0.0 | 1104307.02 | 1104373.33 | regression_outputs/multiple_model_a_summary.txt |
| Pairwise OLS: price ~ wind_speed_10m + generation_forecast | price | pairwise | wind_speed_10m + generation_forecast | 2 | wind_speed_10m, generation_forecast | 96133 | 204 | 0.0797 | 0.0796 | 0.0797 | 0.0 | 1104580.01 | 1104608.43 | regression_outputs/pair_wind_speed_10m__generation_forecast_summary.txt |
| Pairwise OLS: price ~ wind_speed_10m + shortwave_radiation | price | pairwise | wind_speed_10m + shortwave_radiation | 2 | wind_speed_10m, shortwave_radiation | 96337 | 0 | 0.0461 | 0.0461 | 0.0461 | 0.0 | 1110215.4 | 1110243.83 | regression_outputs/pair_wind_speed_10m__shortwave_radiation_summary.txt |
| Pairwise OLS: price ~ cloud_cover + wind_speed_10m | price | pairwise | cloud_cover + wind_speed_10m | 2 | cloud_cover, wind_speed_10m | 96337 | 0 | 0.0385 | 0.0385 | 0.0385 | 0.0 | 1110981.73 | 1111010.16 | regression_outputs/pair_cloud_cover__wind_speed_10m_summary.txt |
| Pairwise OLS: price ~ temperature_2m + wind_speed_10m | price | pairwise | temperature_2m + wind_speed_10m | 2 | temperature_2m, wind_speed_10m | 96337 | 0 | 0.0381 | 0.038 | 0.0381 | 0.0 | 1111027.6 | 1111056.02 | regression_outputs/pair_temperature_2m__wind_speed_10m_summary.txt |

The strongest simple regressions are still limited as forecasting evidence. They show measurable linear association, but the residual diagnostics confirm that a linear contemporaneous model does not explain the full price dynamics, especially during high-volatility periods and extreme events.

### Single-variable price regressions

This table answers requests such as `total_load vs price`, `temperature_2m vs price`, and `wind_speed_10m vs price` directly. Each row is one separate `price ~ predictor` model.

| primary_feature | r2 | adj_r2 | explained_variance | model_pvalue | nobs |
| --- | --- | --- | --- | --- | --- |
| wind_speed_10m | 0.0368 | 0.0368 | 0.0368 | 0.0 | 96337 |
| shortwave_radiation | 0.0106 | 0.0106 | 0.0106 | 2.6914597519202063e-226 | 96337 |
| temperature_2m | 0.0009 | 0.0008 | 0.0009 | 9.40440893167978e-20 | 96337 |
| generation_forecast | 0.0005 | 0.0005 | 0.0005 | 9.99943847895289e-13 | 96133 |
| total_load | 0.0003 | 0.0003 | 0.0003 | 9.254203583799459e-08 | 96326 |
| cloud_cover | 0.0 | -0.0 | 0.0 | 0.8243695288125994 | 96337 |

#### Plain-language interpretation of the single-feature models

| predictor | direction | statistically_significant_0_05 | r2 | explained_variance | interpretation_note |
| --- | --- | --- | --- | --- | --- |
| wind_speed_10m | falls | True | 0.0368 | 0.0368 | Statistically significant with some explanatory value, but still only an in-sample linear screen. |
| shortwave_radiation | falls | True | 0.0106 | 0.0106 | Statistically significant with some explanatory value, but still only an in-sample linear screen. |
| temperature_2m | falls | True | 0.0009 | 0.0009 | Statistically significant, but the model explains almost none of the price variance. |
| generation_forecast | falls | True | 0.0005 | 0.0005 | Statistically significant, but the model explains almost none of the price variance. |
| total_load | falls | True | 0.0003 | 0.0003 | Statistically significant, but the model explains almost none of the price variance. |
| cloud_cover | falls | False | 0.0 | 0.0 | Not statistically significant in this simple linear screen. |

For `total_load`, the fitted coefficient is **-0.000547**. That means the simple same-time model estimates that as load rises, fitted price **falls** slightly on average, not rises. The model p-value is **9.25e-08**, so the relationship is statistically distinguishable from zero under the OLS assumptions, but the `R²` is only **0.0003**. In practical terms, that means the effect is statistically detectable in a very large sample while explaining almost none of the price variance.

`p < 0.05` should therefore be read as **evidence of a non-zero linear association in-sample**, not as proof that the feature should definitely be included in a forecasting model. Inclusion still depends on forecast-time availability, leakage risk, stability across time, and out-of-sample usefulness.

![Single-variable price regression ranking](figures/static/price_single_variable_regression_r2_ranking.png)

### Two-variable price regressions

This table covers specifications such as `cloud_cover + wind_speed_10m vs price` and every other two-feature combination from the current exogenous shortlist. The best two-variable combination in this screen is **wind_speed_10m + generation_forecast with adjusted R² 0.0796**.

| primary_feature | r2 | adj_r2 | explained_variance | model_pvalue | nobs |
| --- | --- | --- | --- | --- | --- |
| wind_speed_10m + generation_forecast | 0.0797 | 0.0796 | 0.0797 | 0.0 | 96133 |
| wind_speed_10m + shortwave_radiation | 0.0461 | 0.0461 | 0.0461 | 0.0 | 96337 |
| cloud_cover + wind_speed_10m | 0.0385 | 0.0385 | 0.0385 | 0.0 | 96337 |
| temperature_2m + wind_speed_10m | 0.0381 | 0.038 | 0.0381 | 0.0 | 96337 |
| wind_speed_10m + total_load | 0.0369 | 0.0368 | 0.0369 | 0.0 | 96326 |
| shortwave_radiation + total_load | 0.0123 | 0.0122 | 0.0123 | 1.4427981294166306e-258 | 96326 |
| shortwave_radiation + generation_forecast | 0.0123 | 0.0122 | 0.0123 | 5.054012048462595e-258 | 96133 |
| temperature_2m + shortwave_radiation | 0.0116 | 0.0116 | 0.0116 | 6.639588162949627e-245 | 96337 |
| cloud_cover + shortwave_radiation | 0.0109 | 0.0109 | 0.0109 | 3.8947624302978227e-230 | 96337 |
| temperature_2m + total_load | 0.0015 | 0.0015 | 0.0015 | 3.160031937121078e-32 | 96326 |
| temperature_2m + generation_forecast | 0.0016 | 0.0015 | 0.0016 | 2.1458644535858492e-33 | 96133 |
| temperature_2m + cloud_cover | 0.0009 | 0.0008 | 0.0009 | 7.753069973516902e-19 | 96337 |
| total_load + generation_forecast | 0.0008 | 0.0008 | 0.0008 | 1.1759168990337963e-17 | 96133 |
| cloud_cover + generation_forecast | 0.0005 | 0.0005 | 0.0005 | 5.9468979356321535e-12 | 96133 |
| cloud_cover + total_load | 0.0003 | 0.0003 | 0.0003 | 6.286711811929127e-07 | 96326 |

| model_features | adj_r2 | feature_effects | interpretation_note |
| --- | --- | --- | --- |
| wind_speed_10m + generation_forecast | 0.0796 | wind_speed_10m: price falls when the variable rises (significant, coef=-5.3764, p=0) | generation_forecast: price rises when the variable rises (significant, coef=0.0253, p=0) | Overall model is statistically significant with moderate explanatory power. |
| wind_speed_10m + shortwave_radiation | 0.0461 | wind_speed_10m: price falls when the variable rises (significant, coef=-2.2190, p=0) | shortwave_radiation: price falls when the variable rises (significant, coef=-0.0388, p=5.55e-205) | Overall model is statistically significant with weak explanatory power. |
| cloud_cover + wind_speed_10m | 0.0385 | cloud_cover: price rises when the variable rises (significant, coef=10.3051, p=3.28e-38) | wind_speed_10m: price falls when the variable rises (significant, coef=-2.3639, p=0) | Overall model is statistically significant with weak explanatory power. |
| temperature_2m + wind_speed_10m | 0.038 | temperature_2m: price falls when the variable rises (significant, coef=-0.4148, p=3.5e-28) | wind_speed_10m: price falls when the variable rises (significant, coef=-2.2714, p=0) | Overall model is statistically significant with weak explanatory power. |
| wind_speed_10m + total_load | 0.0368 | wind_speed_10m: price falls when the variable rises (significant, coef=-2.2555, p=0) | total_load: price falls when the variable rises (not significant, coef=-0.0002, p=0.0801) | Overall model is statistically significant with weak explanatory power. |
| shortwave_radiation + total_load | 0.0122 | shortwave_radiation: price falls when the variable rises (significant, coef=-0.0450, p=5.3e-254) | total_load: price falls when the variable rises (significant, coef=-0.0013, p=1.11e-35) | Overall model is statistically significant with weak explanatory power. |
| shortwave_radiation + generation_forecast | 0.0122 | shortwave_radiation: price falls when the variable rises (significant, coef=-0.0441, p=1.32e-248) | generation_forecast: price falls when the variable rises (significant, coef=-0.0030, p=7.01e-35) | Overall model is statistically significant with weak explanatory power. |
| temperature_2m + shortwave_radiation | 0.0116 | temperature_2m: price rises when the variable rises (significant, coef=0.4365, p=5e-22) | shortwave_radiation: price falls when the variable rises (significant, coef=-0.0494, p=1.52e-228) | Overall model is statistically significant with weak explanatory power. |

![Two-variable price regression ranking](figures/static/price_two_variable_regression_r2_ranking.png)

![Multi-feature model diagnostics](figures/static/multiple_model_b_diagnostics.png)

![Multi-feature standardized coefficients](figures/static/multiple_model_b_standardized_coefficients.png)

## 8. Null, Noise, and Anomaly Analysis

Weaknesses are concentrated in three areas:

- missing blocks in the system variables, especially `generation_forecast`
- regime changes and price spikes that increase residual error and heavy tails
- weather variables whose direct linear relationship with price is weaker than the system variables

![Price anomalies](figures/static/anomaly_candidates_price.png)

![Generation anomalies](figures/static/anomaly_candidates_generation_forecast.png)

![Load anomalies](figures/static/anomaly_candidates_total_load.png)

![Outlier comparison](figures/static/outlier_comparison.png)

These issues are not severe enough to make the dataset unusable, but they are severe enough to bias naive models and random train/test splits.

## 9. Forecast-time Availability and Leakage Risks

| feature | category | reason |
| --- | --- | --- |
| hour, day_of_week, month, is_weekend, hour_sin, hour_cos | safe | Calendar features are known when the forecast is issued. |
| lagged price features based only on past observations | safe | Past prices are available at forecast time. |
| rolling features computed only from past windows | safe | Past-only windows do not leak future target information. |
| temperature_2m, cloud_cover, wind_speed_10m, shortwave_radiation | safe only if forecasted | Observed future weather is not available for a 48-hour-ahead forecast unless forecast data is used. |
| generation_forecast | unclear | The name suggests a forecast, but the dataset does not include forecast issuance timestamps or vintages. |
| total_load | safe only if forecasted | Observed future load would leak; forecasted load could be safe if only the ex-ante forecast is used. |
| future observed production or weather values aligned directly with the target horizon | unsafe / leaky | These values would not be known when issuing the forecast. |
| centered rolling windows or features derived from future prices | unsafe / leaky | They use future observations from the prediction period. |

This section is central to later modeling. The current exogenous files do not prove that weather, load, or generation values were available as real ex-ante forecasts at the moment a 48-hour-ahead forecast would be issued. Future observed values must therefore be treated as unavailable unless forecast-vintage data is added.

## 10. Key Findings

- The dataset is operationally mixed-frequency: the target becomes 15-minute in 2025, but exogenous sources remain hourly.
- The target series has strong intraday structure, heavy tails, negative values, and clear regime shifts in volatility.
- The most promising exogenous candidates are currently: **total_load, wind_speed_10m, shortwave_radiation, generation_forecast, temperature_2m**.
- Weaker direct signals in the current panel include: **cloud_cover, temperature_2m**.
- The most visible data-quality issues are concentrated in: **generation_forecast, total_load**.
- The prebuilt clean dataset hides missingness by removing rows, so future modeling should start from the raw aligned panel and apply explicit preprocessing.

## 11. Recommended Next Steps

1. Preserve this hourly aligned panel as the audited benchmark dataset for feature engineering experiments.
2. Build lag features for price and exogenous variables using only past information, with candidate lags informed by the lag-correlation screen.
3. Add rolling features such as trailing 24-hour mean, trailing 24-hour standard deviation, and trailing same-hour historical summaries.
4. Use calendar features in every baseline model.
5. Avoid random splitting. Use chronological train, validation, and test windows with rolling or expanding evaluation.
6. For the true 48-hour / 15-minute objective, acquire or reconstruct forecast-vintage exogenous inputs at matching operational resolution.
7. Benchmark future models against Prophet, but evaluate using time-aware metrics such as MAE, RMSE, and sMAPE or MAPE with care around near-zero values.

## 12. Recommended Figures for Presentation

| figure | what_it_shows | why_it_matters |
| --- | --- | --- |
| price_full_series.png | Long-run target history with volatility shifts and extreme spikes. | Establishes that the target is non-stationary and heavy-tailed. |
| price_frequency_transition.png | The switch from hourly to quarter-hour price resolution at the start of 2025. | Explains why the current dataset cannot yet support a fully aligned 15-minute exogenous panel. |
| missingness_over_time.png | When exogenous missing blocks occur over time. | Supports explicit imputation or exclusion decisions instead of silent row dropping. |
| correlation_heatmap_weather_production_price.png | Initial structure among price, weather, system, and calendar variables. | Quickly highlights candidate features and redundant relationships. |
| lagged_correlation_screen.png | How exogenous-price correlation changes across historical lags. | Guides later lag-feature engineering without overclaiming forecasting value. |
| weather_r2_price_load_comparison.png | How strongly each weather variable explains price versus total load in pairwise OLS screening. | Directly answers whether weather is more linearly tied to demand than to price. |
| price_two_variable_regression_r2_ranking.png | Which two-variable price regressions explain the most variance among all exogenous feature pairs. | Makes pair combinations like `cloud_cover + wind_speed_10m` comparable instead of anecdotal. |
| multiple_model_b_diagnostics.png | Fit quality and residual behavior for the strongest exploratory linear model. | Shows both the usable signal and the remaining modeling gap. |
| multiple_model_b_standardized_coefficients.png | Relative influence of the multi-feature predictors on a comparable scale. | Supports feature selection and interpretation discussions. |
| anomaly_candidates_price.png | Flagged price spikes and unusual low-price periods. | Highlights where robust forecasting methods and careful evaluation will matter most. |
