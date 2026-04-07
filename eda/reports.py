from __future__ import annotations

import pandas as pd

from eda.config import EDA_DIR, TARGET
from eda.utils import markdown_table


def write_report(
    inventory_df: pd.DataFrame,
    notes_df: pd.DataFrame,
    quality_table: pd.DataFrame,
    suspicious_table: pd.DataFrame,
    signal_screen: pd.DataFrame,
    regression_metrics: pd.DataFrame,
    weather_pairwise_metrics: pd.DataFrame,
    weather_r2_summary: pd.DataFrame,
    single_price_interpretations: pd.DataFrame,
    pairwise_price_interpretations: pd.DataFrame,
    leakage_table: pd.DataFrame,
    audit_summary: pd.DataFrame,
    removed_times_df: pd.DataFrame,
    recommended_figures: pd.DataFrame,
    figure_paths: dict[str, str],
    key_numbers: dict,
    recommendations: dict,
) -> None:
    top_corr = signal_screen[["feature", "corr_with_price_t", "best_abs_corr", "best_lag_hours"]].head(6)
    top_models = regression_metrics.sort_values("adj_r2", ascending=False).head(6)
    top_quality_issues = quality_table.sort_values(["missing_pct", "robust_anomaly_count"], ascending=False).head(7)
    single_price_models = regression_metrics[
        (regression_metrics["target_column"] == TARGET) & (regression_metrics["model_type"] == "simple")
    ].sort_values("adj_r2", ascending=False)
    pairwise_price_models = regression_metrics[
        (regression_metrics["target_column"] == TARGET) & (regression_metrics["model_type"] == "pairwise")
    ].sort_values("adj_r2", ascending=False)
    best_pairwise_model = pairwise_price_models.iloc[0] if not pairwise_price_models.empty else None
    best_pairwise_summary = (
        f"{best_pairwise_model['primary_feature']} with adjusted R² {best_pairwise_model['adj_r2']:.4f}"
        if best_pairwise_model is not None
        else "N/A"
    )
    weather_price_top = weather_pairwise_metrics[weather_pairwise_metrics["target_column"] == TARGET].sort_values("r2", ascending=False)
    weather_load_top = weather_pairwise_metrics[weather_pairwise_metrics["target_column"] == "total_load"].sort_values("r2", ascending=False)
    load_interpretation = single_price_interpretations[single_price_interpretations["predictor"] == "total_load"].iloc[0]
    removed_preview = removed_times_df.head(10).copy()
    if not removed_preview.empty:
        removed_preview["time_removed_in_clean"] = removed_preview["time_removed_in_clean"].astype(str)

    report = f"""# EDA Report: EPEX Electricity Price Forecasting

## 1. Introduction

This report documents a reproducible exploratory data analysis for the current electricity price forecasting dataset. The project goal is to forecast EPEX electricity prices 48 hours ahead at 15-minute intervals, but the available exogenous sources are still hourly. The analysis therefore splits into two tracks:

- an **hourly aligned panel** for joint price, production, and weather analysis
- a **2025 15-minute price-only segment** to document the current resolution change in the target series

All timestamps were parsed as naive datetimes. This was deliberate because the repository documentation says UTC, while the load and generation files show daylight-saving spring-forward gaps that are inconsistent with a clean UTC timeline.

## 2. Dataset Overview

### Source inventory

{markdown_table(inventory_df.assign(date_start=inventory_df["date_start"].astype(str), date_end=inventory_df["date_end"].astype(str)))}

### Key inventory notes

{markdown_table(notes_df)}

### Combined-panel audit

{markdown_table(audit_summary)}

The source-based hourly panel keeps nulls for auditability. The prebuilt clean dataset removes rows entirely when exogenous values are absent, which shortens the usable sample and hides the original missing-value pattern.

## 3. Data Alignment and Time Handling

The source data does not align cleanly to a single operational forecasting panel:

- The price series is hourly-only through **{key_numbers["hourly_only_segment_end"]}** and then expands to quarter-hour resolution from **{key_numbers["first_quarter_hour_ts"]}** onward.
- Weather, load, and generation remain hourly across the full period.
- Load and generation show spring-forward gaps at `02:00` on daylight-saving transition dates, while price and weather still contain those timestamps.
- The hourly aligned panel was constructed by keeping only top-of-hour price observations and left-joining weather, load, and generation on exact timestamps.

![Source coverage timeline]({figure_paths["source_coverage_timeline"]})

![Price resolution transition]({figure_paths["price_frequency_transition"]})

![Detected source gaps]({figure_paths["source_gap_events"]})

## 4. Data Quality Findings

### Missingness and stability

{markdown_table(top_quality_issues)}

### Rule-based suspicious values

{markdown_table(suspicious_table)}

### Missing timestamps removed by the prebuilt clean dataset

{markdown_table(removed_preview) if not removed_preview.empty else "_No removed timestamps._"}

The main quality issues are concentrated in the exogenous system variables rather than the target itself. Missingness is dominated by blocks in `generation_forecast`, plus repeated DST-related nulls in `total_load` and `generation_forecast` at the missing `02:00` hour in spring. These rows should be reported and handled explicitly during modeling rather than silently dropped up front.

![Missingness by column]({figure_paths["missingness_bar"]})

![Missingness heatmap]({figure_paths["missingness_heatmap"]})

![Missingness over time]({figure_paths["missingness_over_time"]})

## 5. Target Series Analysis

The hourly target series shows clear intraday structure, changing volatility, and episodic spikes. Negative prices are present, so models must preserve them rather than clipping them away. The distribution is skewed and heavy-tailed, and the volatility profile changes materially over time, which argues against assuming a stationary low-variance process.

- Negative hourly prices observed: **{key_numbers["negative_price_count"]}**
- Hourly price mean: **{key_numbers["price_mean"]:.2f} EUR/MWh**
- Hourly price standard deviation: **{key_numbers["price_std"]:.2f} EUR/MWh**
- Hourly price min / max: **{key_numbers["price_min"]:.2f} / {key_numbers["price_max"]:.2f} EUR/MWh**

![Full price series]({figure_paths["price_full_series"]})

![Representative month]({figure_paths["price_month_zoom"]})

![Representative week]({figure_paths["price_week_zoom"]})

![Representative day]({figure_paths["price_day_zoom"]})

![Price distribution]({figure_paths["price_distribution"]})

![Price by hour]({figure_paths["price_boxplot_hour"]})

![Price by day of week]({figure_paths["price_boxplot_dow"]})

![Rolling mean]({figure_paths["price_rolling_mean"]})

![Rolling volatility]({figure_paths["price_rolling_std"]})

![Weekday vs weekend]({figure_paths["price_weekday_weekend"]})

![Price ACF]({figure_paths["price_acf"]})

![Price PACF]({figure_paths["price_pacf"]})

![Seasonal decomposition]({figure_paths["price_decomposition"]})

### 2025 15-minute price-only segment

The 2025 target series provides the quarter-hour resolution required by the project goal, but only for the target. Because the exogenous sources remain hourly, this section is descriptive only and should not be mistaken for a fully aligned 15-minute forecasting panel.

![2025 15-minute full series]({figure_paths["price_2025_full"]})

![2025 month zoom]({figure_paths["price_2025_month"]})

![2025 week zoom]({figure_paths["price_2025_week"]})

![2025 day zoom]({figure_paths["price_2025_day"]})

![2025 distribution]({figure_paths["price_2025_distribution"]})

## 6. Weather and Production Feature Analysis

The exogenous variables are informative but not equally strong. The system-level variables tend to have a clearer relationship with price than the raw weather variables, while weather remains useful as a causal driver for renewable output and demand conditions.

### Initial feature screening

{markdown_table(top_corr)}

### Signal-quality screen

{markdown_table(signal_screen.head(6))}

![Weather over time]({figure_paths["weather_timeseries"]})

![Production over time]({figure_paths["production_timeseries"]})

![Weather distributions]({figure_paths["weather_histograms"]})

![Production distributions]({figure_paths["production_histograms"]})

![Correlation heatmap]({figure_paths["correlation_heatmap"]})

![Scatter screening]({figure_paths["scatter_grid"]})

![Lagged correlations]({figure_paths["lagged_correlations"]})

The lagged-correlation screen is only an exploratory ranking tool. It helps identify plausible variables and lags for later forecasting features, but it does not prove predictive value at a 48-hour horizon.

### Weather vs Price and Load R² Screening

The teammate request for weather-to-price and weather-to-load `R²` is implemented here as **pairwise simple OLS**. Each model uses one weather variable at a time and a single target. In this context:

- **R²** measures how much same-time variance in the target is explained by one linear weather predictor.
- **Model p-value** tests whether the overall fitted linear relationship is distinguishable from zero under the OLS assumptions.
- **Explained variance** is included alongside `R²` because it gives another summary of how much fitted variation the model captures, using the model predictions directly.

These statistics are useful for screening and practice, but they are **not forecasting metrics**. A low pairwise `R²` does not mean the feature is useless later, because lagged effects, interactions, nonlinear models, and joint modeling can still make the feature valuable.

#### Pairwise weather OLS ranking

{markdown_table(weather_r2_summary[["feature", "r2_price", "r2_total_load", "delta_r2_load_minus_price", "recommendation"]])}

#### Best weather predictors for price

{markdown_table(weather_price_top[["feature", "r2", "adj_r2", "explained_variance", "model_pvalue", "nobs"]].head(4))}

#### Best weather predictors for total load

{markdown_table(weather_load_top[["feature", "r2", "adj_r2", "explained_variance", "model_pvalue", "nobs"]].head(4))}

![Weather R² vs price]({figure_paths["weather_r2_vs_price"]})

![Weather R² vs total load]({figure_paths["weather_r2_vs_load"]})

![Weather R² comparison]({figure_paths["weather_r2_comparison"]})

## 7. Exploratory Regression Analysis

Ordinary least squares was used here as an interpretation tool, not as the final forecasting benchmark. The models were fit on the hourly aligned panel with per-model row dropping where needed.

### Regression metrics

{markdown_table(top_models)}

The strongest simple regressions are still limited as forecasting evidence. They show measurable linear association, but the residual diagnostics confirm that a linear contemporaneous model does not explain the full price dynamics, especially during high-volatility periods and extreme events.

### Single-variable price regressions

This table answers requests such as `total_load vs price`, `temperature_2m vs price`, and `wind_speed_10m vs price` directly. Each row is one separate `price ~ predictor` model.

{markdown_table(single_price_models[["primary_feature", "r2", "adj_r2", "explained_variance", "model_pvalue", "nobs"]])}

#### Plain-language interpretation of the single-feature models

{markdown_table(single_price_interpretations[["predictor", "direction", "statistically_significant_0_05", "r2", "explained_variance", "interpretation_note"]])}

For `total_load`, the fitted coefficient is **{load_interpretation["coefficient"]:.6f}**. That means the simple same-time model estimates that as load rises, fitted price **{load_interpretation["direction"]}** slightly on average, not rises. The model p-value is **{load_interpretation["model_pvalue"]:.3g}**, so the relationship is statistically distinguishable from zero under the OLS assumptions, but the `R²` is only **{load_interpretation["r2"]:.4f}**. In practical terms, that means the effect is statistically detectable in a very large sample while explaining almost none of the price variance.

`p < 0.05` should therefore be read as **evidence of a non-zero linear association in-sample**, not as proof that the feature should definitely be included in a forecasting model. Inclusion still depends on forecast-time availability, leakage risk, stability across time, and out-of-sample usefulness.

![Single-variable price regression ranking]({figure_paths["price_single_variable_regression_ranking"]})

### Two-variable price regressions

This table covers specifications such as `cloud_cover + wind_speed_10m vs price` and every other two-feature combination from the current exogenous shortlist. The best two-variable combination in this screen is **{best_pairwise_summary}**.

{markdown_table(pairwise_price_models[["primary_feature", "r2", "adj_r2", "explained_variance", "model_pvalue", "nobs"]]) if not pairwise_price_models.empty else "_No pairwise price models were generated._"}

{markdown_table(pairwise_price_interpretations[["model_features", "adj_r2", "feature_effects", "interpretation_note"]].head(8)) if not pairwise_price_interpretations.empty else "_No pairwise price interpretation table was generated._"}

![Two-variable price regression ranking]({figure_paths["price_pairwise_regression_ranking"]})

![Multi-feature model diagnostics]({figure_paths["regression_multi_b_diagnostics"]})

![Multi-feature standardized coefficients]({figure_paths["regression_multi_b_coefficients"]})

## 8. Null, Noise, and Anomaly Analysis

Weaknesses are concentrated in three areas:

- missing blocks in the system variables, especially `generation_forecast`
- regime changes and price spikes that increase residual error and heavy tails
- weather variables whose direct linear relationship with price is weaker than the system variables

![Price anomalies]({figure_paths["anomaly_price"]})

![Generation anomalies]({figure_paths["anomaly_generation"]})

![Load anomalies]({figure_paths["anomaly_load"]})

![Outlier comparison]({figure_paths["outlier_boxplots"]})

These issues are not severe enough to make the dataset unusable, but they are severe enough to bias naive models and random train/test splits.

## 9. Forecast-time Availability and Leakage Risks

{markdown_table(leakage_table)}

This section is central to later modeling. The current exogenous files do not prove that weather, load, or generation values were available as real ex-ante forecasts at the moment a 48-hour-ahead forecast would be issued. Future observed values must therefore be treated as unavailable unless forecast-vintage data is added.

## 10. Key Findings

- The dataset is operationally mixed-frequency: the target becomes 15-minute in 2025, but exogenous sources remain hourly.
- The target series has strong intraday structure, heavy tails, negative values, and clear regime shifts in volatility.
- The most promising exogenous candidates are currently: **{recommendations["promising"]}**.
- Weaker direct signals in the current panel include: **{recommendations["weak"]}**.
- The most visible data-quality issues are concentrated in: **{recommendations["missing"]}**.
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

{markdown_table(recommended_figures)}
"""
    (EDA_DIR / "eda_report.md").write_text(report, encoding="utf-8")


def write_findings_summary(
    recommendations: dict,
    key_numbers: dict,
    weather_r2_summary: pd.DataFrame,
    regression_metrics: pd.DataFrame,
    single_price_interpretations: pd.DataFrame,
) -> None:
    price_top = weather_r2_summary.sort_values("r2_price", ascending=False).iloc[0]
    load_top = weather_r2_summary.sort_values("r2_total_load", ascending=False).iloc[0]
    best_model = regression_metrics.sort_values("adj_r2", ascending=False).iloc[0]
    best_single_model = regression_metrics[
        (regression_metrics["target_column"] == TARGET) & (regression_metrics["model_type"] == "simple")
    ].sort_values("adj_r2", ascending=False).iloc[0]
    best_pair_model = regression_metrics[
        (regression_metrics["target_column"] == TARGET) & (regression_metrics["model_type"] == "pairwise")
    ].sort_values("adj_r2", ascending=False).iloc[0]
    load_row = single_price_interpretations[single_price_interpretations["predictor"] == "total_load"].iloc[0]
    relationship_line = (
        "Weather appears more directly tied to load than to price in pairwise linear screening."
        if weather_r2_summary["delta_r2_load_minus_price"].mean() > 0
        else "Weather does not explain load more strongly than price on average in pairwise linear screening."
    )

    text = f"""# Findings Summary

## Core findings

- Negative hourly prices: **{key_numbers["negative_price_count"]}**
- Hourly price range: **{key_numbers["price_min"]:.2f} to {key_numbers["price_max"]:.2f} EUR/MWh**
- Target frequency break: hourly through **{key_numbers["hourly_only_segment_end"]}**, then quarter-hour from **{key_numbers["first_quarter_hour_ts"]}**
- Strongest multi-feature exploratory OLS model: **{best_model["model_name"]}** with adjusted R² **{best_model["adj_r2"]:.4f}**

## Top weather variables by pairwise R²

- Best for price: **{price_top["feature"]}** with R² **{price_top["r2_price"]:.4f}**
- Best for total load: **{load_top["feature"]}** with R² **{load_top["r2_total_load"]:.4f}**
- Interpretation: {relationship_line}

## Variable-level price regressions

- Best single-variable price model: **{best_single_model["primary_feature"]}** with adjusted R² **{best_single_model["adj_r2"]:.4f}**
- Best two-variable price model: **{best_pair_model["primary_feature"]}** with adjusted R² **{best_pair_model["adj_r2"]:.4f}**
- The by-variable regression outputs now explicitly cover every single exogenous feature and every two-feature combination against price.
- `total_load -> price`: fitted direction is **{load_row["direction"]}**, model p-value is **{load_row["model_pvalue"]:.3g}**, but R² is only **{load_row["r2"]:.4f}**, so the effect is statistically detectable and still very weak in explanatory terms.

## Modeling implications

- Most promising current candidate features: **{recommendations["promising"]}**
- Weak direct signals in the current hourly panel: **{recommendations["weak"]}**
- Most visible missingness issues: **{recommendations["missing"]}**
- Pairwise weather R² should be used for explanatory screening only, not as a substitute for time-aware forecast validation.
"""
    (EDA_DIR / "findings_summary.md").write_text(text, encoding="utf-8")


def write_technical_notes() -> None:
    text = """# Technical Notes

## Framework stack

- `pandas` for loading CSV files, joining datasets, grouping, missingness tables, correlation tables, and CSV export
- `numpy` for numerical transforms and calendar encodings
- `matplotlib` and `seaborn` for static report-ready figures
- `plotly.express` and `plotly.io` for interactive HTML charts
- `statsmodels` for OLS, OLS summaries, ACF, PACF, seasonal decomposition, and Q-Q plots
- `scikit-learn` for explained variance

## How the tables are created

- Most tables are built as `pandas.DataFrame` objects and written to `eda/tables/` with `DataFrame.to_csv(...)`.
- Markdown tables inside `eda_report.md` are generated by the helper `markdown_table(df)` in `eda/eda.py`.
- Regression coefficient tables come from the fitted `statsmodels.OLS` object and are saved to `eda/regression_outputs/*_coefficients.csv`.
- Fitted values and residuals are saved to `eda/regression_outputs/*_fitted_values.csv`.

## Pairwise weather OLS screening

The weather-to-price and weather-to-load section uses simple one-predictor OLS models:

- `price ~ temperature_2m`
- `price ~ cloud_cover`
- `price ~ wind_speed_10m`
- `price ~ shortwave_radiation`
- `total_load ~ temperature_2m`
- `total_load ~ cloud_cover`
- `total_load ~ wind_speed_10m`
- `total_load ~ shortwave_radiation`

Rows are dropped per model only if the selected predictor or target is null. Each model is fit with an intercept using `statsmodels.api.OLS` after `sm.add_constant(...)`.

## Variable-level price regressions

The script now also generates an explicit by-variable regression layer for the price target:

- all six single-feature models of the form `price ~ x`
- all fifteen two-feature models of the form `price ~ x1 + x2`

The two-feature models are generated programmatically with `itertools.combinations(EXOGENOUS_COLUMNS, 2)`. This makes examples such as `total_load vs price`, `temperature_2m vs price`, and `cloud_cover + wind_speed_10m vs price` reproducible instead of ad hoc.

## Regression metrics on the plots

The shared regression helpers now read metrics directly from the fitted statsmodels result:

- `R²` from `model.rsquared`
- `Adjusted R²` from `model.rsquared_adj`
- model-level `p-value` from `model.f_pvalue`
- explained variance from `sklearn.metrics.explained_variance_score(y_true, y_pred)`

The helpers that place those metrics onto figures are:

- `plot_simple_regression(...)`
- `plot_regression_diagnostics(...)`
- `plot_weather_r2_bars(...)`
- `plot_weather_r2_comparison(...)`

This keeps the numeric annotations consistent between the saved summaries, tables, and plots.

## Interpreting p-values in this project

- `p < 0.05` means the fitted linear relation is statistically distinguishable from zero under the model assumptions.
- It does **not** automatically mean the feature should definitely be included in a forecasting model.
- In this dataset, the sample is large enough that very small effects can produce tiny p-values while still having almost no explanatory power.
- For time-series forecasting, feature inclusion also depends on leakage risk, forecast-time availability, temporal stability, and out-of-sample validation.
"""
    (EDA_DIR / "technical_notes.md").write_text(text, encoding="utf-8")


def write_documentation_addendum(
    weather_r2_summary: pd.DataFrame,
    regression_metrics: pd.DataFrame,
) -> None:
    best_weather_price = weather_r2_summary.sort_values("r2_price", ascending=False).iloc[0]["feature"]
    best_weather_load = weather_r2_summary.sort_values("r2_total_load", ascending=False).iloc[0]["feature"]
    best_model = regression_metrics.sort_values("adj_r2", ascending=False).iloc[0]["model_name"]
    best_pair_model = regression_metrics[
        (regression_metrics["target_column"] == TARGET) & (regression_metrics["model_type"] == "pairwise")
    ].sort_values("adj_r2", ascending=False).iloc[0]["primary_feature"]
    text = f"""# Documentation Addendum

## What to read first

1. Read `eda/eda_report.md` for the full narrative.
2. Use `eda/findings_summary.md` for the short project-facing summary.
3. Use `eda/tables/` when you need exact numbers for slides or follow-up modeling.
4. Use `eda/regression_outputs/` when you need the raw statsmodels summaries.

## Which outputs answer which questions

- `eda/tables/quality_checks.csv`: which columns are missing, noisy, or anomalous
- `eda/tables/feature_signal_screen.csv`: which exogenous variables look promising in same-time and lagged screening
- `eda/tables/weather_r2_price_load.csv`: whether weather looks more linearly related to price or total load
- `eda/tables/weather_pairwise_ols_metrics.csv`: full pairwise OLS metrics, confidence intervals, and sample sizes
- `eda/tables/price_single_variable_regression_metrics.csv`: direct one-feature `price ~ x` model comparison
- `eda/tables/price_two_variable_regression_metrics.csv`: direct two-feature `price ~ x1 + x2` model comparison
- `eda/regression_outputs/`: raw OLS summaries for practice and diagnostics review

## What we know with higher confidence

- The target has a genuine frequency change from hourly to quarter-hour in 2025.
- The exogenous sources remain hourly and have operational availability risks.
- System variables are currently stronger direct signals than raw weather in same-time screening.
- The strongest weather screen differs by target: `{best_weather_price}` for price and `{best_weather_load}` for load.
- The strongest two-variable price screen currently uses `{best_pair_model}`.

## What remains uncertain

- Whether `generation_forecast` is operationally safe without forecast-vintage timestamps
- How much of the weak pairwise weather signal becomes useful once lags and nonlinearities are included
- How much performance is achievable at the true 48-hour / 15-minute forecasting horizon with operationally valid exogenous data

## How to interpret the teammate R² request

The new weather-vs-price and weather-vs-load section is an explanatory screening exercise. It is useful because it shows whether weather looks more directly connected to demand than to price under simple linear fits. It is not a substitute for forecast evaluation. The broader regression section still matters because the best current exploratory model is `{best_model}`, which shows that multiple features and calendar controls capture more structure than any single weather variable alone.
"""
    (EDA_DIR / "documentation_addendum.md").write_text(text, encoding="utf-8")


def write_regression_interpretation_notes(
    single_price_interpretations: pd.DataFrame,
    pairwise_price_interpretations: pd.DataFrame,
) -> None:
    load_row = single_price_interpretations[single_price_interpretations["predictor"] == "total_load"].iloc[0]
    temp_row = single_price_interpretations[single_price_interpretations["predictor"] == "temperature_2m"].iloc[0]
    wind_row = single_price_interpretations[single_price_interpretations["predictor"] == "wind_speed_10m"].iloc[0]
    cloud_wind_row = pairwise_price_interpretations[pairwise_price_interpretations["model_features"] == "cloud_cover + wind_speed_10m"].iloc[0]

    text = f"""# Regression Interpretation Notes

## Direct answers to common questions

### Does price rise when load rises?

No, not in the simple same-time linear screen. The fitted coefficient in `price ~ total_load` is **{load_row["coefficient"]:.6f}**, so as load rises, the fitted same-time price **{load_row["direction"]}** slightly on average. The model p-value is **{load_row["model_pvalue"]:.3g}**, which is below 0.05, but the R² is only **{load_row["r2"]:.4f}**. That means the relationship is statistically detectable yet explains almost none of the price variation.

### Does price rise when temperature rises?

No in the simple same-time model. The fitted coefficient in `price ~ temperature_2m` is **{temp_row["coefficient"]:.4f}**, so as temperature rises, fitted price **{temp_row["direction"]}** on average. The relationship is statistically significant, but the explanatory power is still very weak with R² **{temp_row["r2"]:.4f}**.

### Does price rise when wind speed rises?

No in the simple same-time model. The fitted coefficient in `price ~ wind_speed_10m` is **{wind_row["coefficient"]:.4f}**, so as wind speed rises, fitted price **{wind_row["direction"]}** on average. This is the strongest single-variable price model in the current screen, but its R² is still only **{wind_row["r2"]:.4f}**.

### What about cloud cover and wind speed together?

For `price ~ cloud_cover + wind_speed_10m`, the overall model has adjusted R² **{cloud_wind_row["adj_r2"]:.4f}**. The feature effects are:

- {cloud_wind_row["feature_effects"].split(" | ")[0]}
- {cloud_wind_row["feature_effects"].split(" | ")[1]}

This pair improves on `wind_speed_10m` alone, but it is still an explanatory screen rather than a forecasting evaluation.

## How to interpret p-value and R² correctly

- `p < 0.05` means the fitted linear association is unlikely to be exactly zero under the OLS assumptions.
- It does **not** mean the variable should definitely be included.
- `R²` shows how much in-sample variance is explained by the model. A very small `R²` means the model can be statistically significant and still explain very little.
- For forecasting, a feature should be judged by availability at forecast time, leakage safety, stability over time, and out-of-sample performance, not by p-value alone.
"""
    (EDA_DIR / "regression_interpretation_notes.md").write_text(text, encoding="utf-8")
