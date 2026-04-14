from __future__ import annotations

from itertools import combinations

import pandas as pd
import plotly.express as px

from eda.config import (
    CALENDAR_COLUMNS,
    EDA_DIR,
    EXOGENOUS_COLUMNS,
    FIG_INTERACTIVE_DIR,
    FIG_STATIC_DIR,
    PRODUCTION_COLUMNS,
    REFERENCE_FILES,
    REGRESSION_DIR,
    SOURCE_FILES,
    TABLE_DIR,
    TARGET,
    WEATHER_COLUMNS,
)
from eda.plotting import (
    plot_acf_pacf,
    plot_anomalies,
    plot_box,
    plot_correlation_heatmap,
    plot_feature_histograms,
    plot_gap_events,
    plot_histogram,
    plot_lagged_correlations,
    plot_line,
    plot_missingness_bar,
    plot_missingness_heatmap,
    plot_missingness_over_time,
    plot_outlier_boxplots,
    plot_price_frequency_transition,
    plot_production_timeseries,
    plot_scatter_grid,
    plot_seasonal_decomposition,
    plot_source_coverage,
    plot_weather_timeseries,
    plot_weekday_weekend,
)
from eda.regression import (
    build_price_regression_interpretations,
    build_regression_metric_row,
    feature_combo_label,
    feature_combo_slug,
    fit_ols_model,
    plot_regression_diagnostics,
    plot_regression_model_ranking,
    plot_simple_regression,
    plot_standardized_coefficients,
    plot_weather_r2_bars,
    plot_weather_r2_comparison,
    run_weather_pairwise_screen,
)
from eda.reports import (
    write_documentation_addendum,
    write_findings_summary,
    write_regression_interpretation_notes,
    write_report,
    write_technical_notes,
)
from eda.summary_analysis import (
    audit_joined_datasets,
    build_dataset_inventory,
    build_leakage_table,
    choose_promising_features,
    compute_lagged_correlations,
    data_quality_checks,
    feature_signal_screen,
    recommendation_paragraphs,
    summarize_frequency_transition,
)
from eda.utils import (
    add_calendar_features,
    configure_plotting,
    contiguous_gap_ranges,
    ensure_directories,
    load_csv,
    markdown_table,
    save_plotly_figure,
    select_complete_periods,
    summarize_dataset,
    write_interactive_index,
    write_json,
)


def run_eda() -> None:
    ensure_directories()
    configure_plotting()

    source_frames = {name: load_csv(path) for name, path in SOURCE_FILES.items()}
    reference_frames = {name: load_csv(path) for name, path in REFERENCE_FILES.items()}

    source_summaries = [summarize_dataset(name, SOURCE_FILES[name], df) for name, df in source_frames.items()]
    reference_summaries = [summarize_dataset(name, REFERENCE_FILES[name], df) for name, df in reference_frames.items()]
    price_transition = summarize_frequency_transition(source_frames["price_source"])
    inventory_df, notes_df = build_dataset_inventory(
        source_summaries, reference_summaries, price_transition["first_non_hourly_timestamp"]
    )

    hourly_price = source_frames["price_source"][source_frames["price_source"]["time"].dt.minute == 0].copy()
    hourly_panel = (
        hourly_price.merge(source_frames["weather_source"], on="time", how="left")
        .merge(source_frames["load_source"], on="time", how="left")
        .merge(source_frames["generation_source"], on="time", how="left")
    )
    hourly_panel = add_calendar_features(hourly_panel).sort_values("time").set_index("time")

    price_2025_15m = source_frames["price_source"][source_frames["price_source"]["time"].dt.year == 2025].copy()
    price_2025_15m = add_calendar_features(price_2025_15m).sort_values("time").set_index("time")

    raw_join = reference_frames["final_dataset_full_raw"].copy()
    clean_join = reference_frames["final_dataset_full_clean"].copy()

    gap_rows = []
    for summary, df in zip(source_summaries, source_frames.values()):
        if summary["name"] == "price_source":
            hourly_segment = df[df["time"].dt.minute == 0]
            quarter_segment = df[df["time"].dt.year == 2025]
            for label, segment, expected in [
                ("price_source_hourly_segment", hourly_segment, pd.Timedelta(hours=1)),
                ("price_source_2025_15m_segment", quarter_segment, pd.Timedelta(minutes=15)),
            ]:
                for gap in contiguous_gap_ranges(segment, expected):
                    gap_rows.append(
                        {
                            "source": label,
                            "gap_after": gap["gap_after"],
                            "gap_before": gap["gap_before"],
                            "gap_duration": gap["gap_duration"],
                            "missing_steps": gap["missing_steps"],
                        }
                    )
        else:
            for gap in contiguous_gap_ranges(df, pd.Timedelta(hours=1)):
                gap_rows.append(
                    {
                        "source": summary["name"],
                        "gap_after": gap["gap_after"],
                        "gap_before": gap["gap_before"],
                        "gap_duration": gap["gap_duration"],
                        "missing_steps": gap["missing_steps"],
                    }
                )
    gap_table = pd.DataFrame(gap_rows).sort_values(["source", "gap_after"])

    quality_table, suspicious_table = data_quality_checks(hourly_panel.reset_index())
    lagged_corr = compute_lagged_correlations(hourly_panel, TARGET, EXOGENOUS_COLUMNS, max_lag=48)
    signal_screen = feature_signal_screen(hourly_panel, lagged_corr)
    leakage_table = build_leakage_table()
    audit_summary, removed_times_df = audit_joined_datasets(hourly_panel.reset_index(), raw_join, clean_join)

    inventory_df.to_csv(TABLE_DIR / "dataset_inventory.csv", index=False)
    notes_df.to_csv(TABLE_DIR / "dataset_inventory_notes.csv", index=False)
    gap_table.to_csv(TABLE_DIR / "source_gap_events.csv", index=False)
    quality_table.to_csv(TABLE_DIR / "quality_checks.csv", index=False)
    suspicious_table.to_csv(TABLE_DIR / "suspicious_value_checks.csv", index=False)
    lagged_corr.to_csv(TABLE_DIR / "lagged_correlations.csv", index=False)
    signal_screen.to_csv(TABLE_DIR / "feature_signal_screen.csv", index=False)
    leakage_table.to_csv(TABLE_DIR / "leakage_risk_table.csv", index=False)
    audit_summary.to_csv(TABLE_DIR / "joined_dataset_audit.csv", index=False)
    removed_times_df.to_csv(TABLE_DIR / "rows_removed_in_clean_dataset.csv", index=False)
    hourly_panel.reset_index().to_csv(TABLE_DIR / "hourly_panel_raw.csv", index=False)
    price_2025_15m.reset_index().to_csv(TABLE_DIR / "price_2025_15m.csv", index=False)

    figure_paths: dict[str, str] = {}
    figure_paths["source_coverage_timeline"] = plot_source_coverage(source_summaries, "source_coverage_timeline.png")
    figure_paths["source_gap_events"] = plot_gap_events(gap_table, "source_gap_events.png")
    figure_paths["price_frequency_transition"] = plot_price_frequency_transition(
        source_frames["price_source"].set_index("time"), "price_frequency_transition.png"
    )

    periods_hourly = select_complete_periods(hourly_panel[[TARGET]], freq="H", year_cap=2024)
    figure_paths["price_full_series"] = plot_line(
        hourly_panel, TARGET, "Hourly Electricity Price: Full Series", "Price (EUR/MWh)", "price_full_series.png"
    )
    figure_paths["price_month_zoom"] = plot_line(
        hourly_panel,
        TARGET,
        "Hourly Electricity Price: Representative Month",
        "Price (EUR/MWh)",
        "price_month_zoom.png",
        (periods_hourly["month_start"], periods_hourly["month_end"]),
    )
    figure_paths["price_week_zoom"] = plot_line(
        hourly_panel,
        TARGET,
        "Hourly Electricity Price: Representative Week",
        "Price (EUR/MWh)",
        "price_week_zoom.png",
        (periods_hourly["week_start"], periods_hourly["week_end"]),
    )
    figure_paths["price_day_zoom"] = plot_line(
        hourly_panel,
        TARGET,
        "Hourly Electricity Price: Representative Day",
        "Price (EUR/MWh)",
        "price_day_zoom.png",
        (periods_hourly["day_start"], periods_hourly["day_end"]),
    )
    figure_paths["price_distribution"] = plot_histogram(
        hourly_panel[TARGET], "Distribution of Hourly Electricity Price", "Price (EUR/MWh)", "price_distribution.png"
    )
    figure_paths["price_boxplot_hour"] = plot_box(
        hourly_panel.reset_index(),
        "hour",
        TARGET,
        "Electricity Price by Hour of Day",
        "Hour of day",
        "Price (EUR/MWh)",
        "price_boxplot_by_hour.png",
    )
    figure_paths["price_boxplot_dow"] = plot_box(
        hourly_panel.reset_index(),
        "day_of_week",
        TARGET,
        "Electricity Price by Day of Week",
        "Day of week (0=Mon)",
        "Price (EUR/MWh)",
        "price_boxplot_by_day_of_week.png",
    )
    figure_paths["price_rolling_mean"] = plot_line(
        hourly_panel.assign(price_rolling_mean=hourly_panel[TARGET].rolling(24 * 7, min_periods=24).mean()),
        "price_rolling_mean",
        "7-Day Rolling Mean of Hourly Electricity Price",
        "Rolling mean (EUR/MWh)",
        "price_rolling_mean.png",
    )
    figure_paths["price_rolling_std"] = plot_line(
        hourly_panel.assign(price_rolling_std=hourly_panel[TARGET].rolling(24 * 7, min_periods=24).std()),
        "price_rolling_std",
        "7-Day Rolling Standard Deviation of Hourly Electricity Price",
        "Rolling std (EUR/MWh)",
        "price_rolling_std.png",
    )
    figure_paths["price_weekday_weekend"] = plot_weekday_weekend(hourly_panel.reset_index(), "price_weekday_vs_weekend.png")
    acf_path, pacf_path = plot_acf_pacf(hourly_panel[TARGET], nlags=168, acf_filename="price_acf.png", pacf_filename="price_pacf.png")
    figure_paths["price_acf"] = acf_path
    figure_paths["price_pacf"] = pacf_path
    figure_paths["price_decomposition"] = plot_seasonal_decomposition(
        hourly_panel[TARGET], "price_seasonal_decomposition.png", period=24
    )

    periods_15m = select_complete_periods(price_2025_15m[[TARGET]], freq="15min")
    figure_paths["price_2025_full"] = plot_line(
        price_2025_15m, TARGET, "2025 Quarter-hour Electricity Price: Full Series", "Price (EUR/MWh)", "price_2025_15m_full_series.png"
    )
    figure_paths["price_2025_month"] = plot_line(
        price_2025_15m,
        TARGET,
        "2025 Quarter-hour Electricity Price: Representative Month",
        "Price (EUR/MWh)",
        "price_2025_15m_month_zoom.png",
        (periods_15m["month_start"], periods_15m["month_end"]),
    )
    figure_paths["price_2025_week"] = plot_line(
        price_2025_15m,
        TARGET,
        "2025 Quarter-hour Electricity Price: Representative Week",
        "Price (EUR/MWh)",
        "price_2025_15m_week_zoom.png",
        (periods_15m["week_start"], periods_15m["week_end"]),
    )
    figure_paths["price_2025_day"] = plot_line(
        price_2025_15m,
        TARGET,
        "2025 Quarter-hour Electricity Price: Representative Day",
        "Price (EUR/MWh)",
        "price_2025_15m_day_zoom.png",
        (periods_15m["day_start"], periods_15m["day_end"]),
    )
    figure_paths["price_2025_distribution"] = plot_histogram(
        price_2025_15m[TARGET],
        "Distribution of 2025 Quarter-hour Electricity Price",
        "Price (EUR/MWh)",
        "price_2025_15m_distribution.png",
    )

    figure_paths["missingness_bar"] = plot_missingness_bar(
        hourly_panel.reset_index()[[TARGET] + EXOGENOUS_COLUMNS + CALENDAR_COLUMNS], "missingness_overview.png"
    )
    figure_paths["missingness_heatmap"] = plot_missingness_heatmap(
        hourly_panel[[TARGET] + EXOGENOUS_COLUMNS + CALENDAR_COLUMNS], "missingness_heatmap.png"
    )
    figure_paths["missingness_over_time"] = plot_missingness_over_time(hourly_panel[EXOGENOUS_COLUMNS], "missingness_over_time.png")
    figure_paths["weather_timeseries"] = plot_weather_timeseries(hourly_panel, "weather_timeseries.png")
    figure_paths["production_timeseries"] = plot_production_timeseries(hourly_panel, "production_timeseries.png")
    figure_paths["weather_histograms"] = plot_feature_histograms(
        hourly_panel, WEATHER_COLUMNS, "Weather Feature Distributions", "weather_feature_histograms.png"
    )
    figure_paths["production_histograms"] = plot_feature_histograms(
        hourly_panel, PRODUCTION_COLUMNS, "Production / System Feature Distributions", "production_feature_histograms.png"
    )
    figure_paths["correlation_heatmap"] = plot_correlation_heatmap(
        hourly_panel[[TARGET] + EXOGENOUS_COLUMNS + CALENDAR_COLUMNS], "correlation_heatmap_weather_production_price.png"
    )
    figure_paths["scatter_grid"] = plot_scatter_grid(hourly_panel, TARGET, EXOGENOUS_COLUMNS, "price_vs_exogenous_scatter_grid.png")
    figure_paths["lagged_correlations"] = plot_lagged_correlations(lagged_corr, "lagged_correlation_screen.png")
    figure_paths["anomaly_price"] = plot_anomalies(hourly_panel, TARGET, "anomaly_candidates_price.png")
    figure_paths["anomaly_generation"] = plot_anomalies(hourly_panel, "generation_forecast", "anomaly_candidates_generation_forecast.png")
    figure_paths["anomaly_load"] = plot_anomalies(hourly_panel, "total_load", "anomaly_candidates_total_load.png")
    figure_paths["outlier_boxplots"] = plot_outlier_boxplots(hourly_panel, [TARGET] + EXOGENOUS_COLUMNS, "outlier_comparison.png")

    interactive_charts: list[dict[str, str]] = []

    plotly_fig = px.line(
        hourly_panel.reset_index(),
        x="time",
        y=TARGET,
        title="Interactive Hourly Electricity Price Series",
        labels={"time": "Time", TARGET: "Price (EUR/MWh)"},
    )
    price_full_meta = {
        "title": "Hourly Price Series",
        "description": "Interactive view of the full hourly EPEX-style target series used for the aligned exogenous analysis.",
        "subtitle": "Uses the hourly aligned panel: exact top-of-hour price observations joined with hourly weather, load, and generation.",
        "what_it_is": "A time-series line chart of hourly electricity price across the full available horizon in the audited hourly panel.",
        "data_used": "Column used: price from old/price_2015_2025.csv, restricted to timestamps where minute == 0. This is the target series in hourly_panel_raw.",
        "how_to_read": "The x-axis is time and the y-axis is price in EUR/MWh. Drag to zoom into a period, double-click to reset, and hover to inspect exact timestamps and values.",
        "what_it_shows": "Long-run price behavior, including volatility changes, spikes, crashes, negative-price periods, and structural shifts over time.",
        "why_it_matters": "It shows why forecasting is difficult: the target is not stable, has extreme events, and does not behave like a low-noise stationary process.",
        "notes": [
            "Negative prices are real observations in this dataset and should not be clipped away without a modeling reason.",
            "This page uses only top-of-hour prices so it aligns with hourly exogenous sources.",
        ],
    }
    figure_paths["interactive_price_full"] = save_plotly_figure(plotly_fig, "price_full_series_interactive.html", price_full_meta)
    interactive_charts.append({"filename": "price_full_series_interactive.html", "title": price_full_meta["title"], "description": price_full_meta["description"]})
    plotly_fig = px.line(
        price_2025_15m.reset_index(),
        x="time",
        y=TARGET,
        title="Interactive 2025 Quarter-hour Electricity Price Series",
        labels={"time": "Time", TARGET: "Price (EUR/MWh)"},
    )
    price_15m_meta = {
        "title": "2025 Quarter-hour Price Series",
        "description": "Interactive view of the higher-resolution 2025 target segment.",
        "subtitle": "Uses the native 15-minute target rows from 2025 only. Exogenous hourly variables are not upsampled here.",
        "what_it_is": "A time-series chart of the quarter-hour price series for 2025, which is the part of the target that matches the later project forecasting resolution.",
        "data_used": "Column used: price from old/price_2015_2025.csv for timestamps in calendar year 2025, including minute 00, 15, 30, and 45.",
        "how_to_read": "Read it like the hourly chart, but note that the series is denser. Zoom into weeks or days to see the intraday quarter-hour pattern clearly.",
        "what_it_shows": "The finer-grained target structure available in 2025, including more detailed intraday movements than the hourly panel can show.",
        "why_it_matters": "This confirms that the project target has already moved to 15-minute resolution, while the exogenous sources have not yet caught up.",
        "notes": [
            "This is target-only EDA. It should not be treated as a fully aligned forecasting dataset with exogenous inputs.",
            "The first quarter-hour timestamp in the source data appears on 2025-01-01 00:15:00.",
        ],
    }
    figure_paths["interactive_price_2025_full"] = save_plotly_figure(plotly_fig, "price_2025_15m_full_series_interactive.html", price_15m_meta)
    interactive_charts.append({"filename": "price_2025_15m_full_series_interactive.html", "title": price_15m_meta["title"], "description": price_15m_meta["description"]})

    daily_weather = hourly_panel[WEATHER_COLUMNS].resample("D").mean().reset_index().melt(id_vars="time", var_name="feature", value_name="value")
    plotly_fig = px.line(daily_weather, x="time", y="value", color="feature", title="Interactive Weather Variables Over Time (daily mean)", labels={"time": "Time", "value": "Value", "feature": "Weather feature"})
    weather_meta = {
        "title": "Weather Time Series",
        "description": "Interactive daily-mean weather series from the selected weather source.",
        "subtitle": "Uses temperature_2m, cloud_cover, wind_speed_10m, and shortwave_radiation from old/weather_2015_2025_nl_avg_fixed.csv, aggregated to daily means.",
        "what_it_is": "A multi-line time-series chart for the main weather variables used as candidate exogenous features.",
        "data_used": "Columns used: temperature_2m, cloud_cover, wind_speed_10m, shortwave_radiation after joining into the hourly panel, then resampled to daily mean for readability.",
        "how_to_read": "Each colored line is one weather variable. Toggle lines in the legend to isolate a single variable. Use zoom to inspect seasonal structure or unusual periods.",
        "what_it_shows": "Seasonality, smoother long-run weather trends, and how different variables move over time relative to each other.",
        "why_it_matters": "Weather influences both demand and renewable generation, so this chart helps assess seasonality, coverage, and plausible exogenous signal strength.",
        "notes": [
            "The weather variant with different scaling was intentionally excluded from the main EDA.",
            "Daily aggregation is for interactive readability; the regression and correlation work still uses hourly aligned data.",
        ],
    }
    figure_paths["interactive_weather_timeseries"] = save_plotly_figure(plotly_fig, "weather_timeseries_interactive.html", weather_meta)
    interactive_charts.append({"filename": "weather_timeseries_interactive.html", "title": weather_meta["title"], "description": weather_meta["description"]})

    daily_production = hourly_panel[PRODUCTION_COLUMNS].resample("D").mean().reset_index().melt(id_vars="time", var_name="feature", value_name="value")
    plotly_fig = px.line(daily_production, x="time", y="value", color="feature", title="Interactive Production / System Variables Over Time (daily mean)", labels={"time": "Time", "value": "Value", "feature": "System feature"})
    production_meta = {
        "title": "Production and Load Time Series",
        "description": "Interactive daily-mean system series for demand and forecasted generation.",
        "subtitle": "Uses total_load and generation_forecast from the hourly aligned panel, aggregated to daily means.",
        "what_it_is": "A two-line time-series chart for the system-level variables that are most directly tied to market conditions.",
        "data_used": "Columns used: total_load from old/load_2015_2025.csv and generation_forecast from old/generation_2015_2025.csv after exact timestamp alignment with the hourly panel.",
        "how_to_read": "Compare the shapes and seasonal patterns of the two lines. Zoom into missing periods or unusual episodes to inspect local behavior.",
        "what_it_shows": "Demand and forecasted generation patterns over time, including missing blocks and broad system-level shifts.",
        "why_it_matters": "These variables are usually closer to the price formation mechanism than raw weather alone, so they are important candidate exogenous drivers.",
        "notes": [
            "generation_forecast contains the largest missing blocks in the exogenous data.",
            "These values are treated as exploratory inputs only; operational forecast-time availability is discussed separately in the report.",
        ],
    }
    figure_paths["interactive_production_timeseries"] = save_plotly_figure(plotly_fig, "production_timeseries_interactive.html", production_meta)
    interactive_charts.append({"filename": "production_timeseries_interactive.html", "title": production_meta["title"], "description": production_meta["description"]})

    corr_df = hourly_panel[[TARGET] + EXOGENOUS_COLUMNS + CALENDAR_COLUMNS].corr(numeric_only=True)
    plotly_fig = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Interactive Correlation Heatmap")
    corr_meta = {
        "title": "Correlation Heatmap",
        "description": "Interactive correlation matrix for price, exogenous variables, and calendar controls.",
        "subtitle": "Uses the hourly aligned panel and Pearson correlations across numeric columns.",
        "what_it_is": "A heatmap where each cell shows the linear correlation between two variables in the hourly panel.",
        "data_used": "Columns used: price, temperature_2m, cloud_cover, wind_speed_10m, shortwave_radiation, total_load, generation_forecast, hour_sin, hour_cos, day_of_week, month, is_weekend.",
        "how_to_read": "Redder colors indicate more positive correlation and bluer colors indicate more negative correlation. Values near zero mean weak linear association. Hover to inspect the exact pair and coefficient.",
        "what_it_shows": "Which variables move together, which ones move in opposite directions, and which variables may be redundant or weakly related.",
        "why_it_matters": "This is an efficient first-pass feature-screening tool, but it should not be treated as proof of forecasting value or causality.",
        "notes": [
            "The matrix uses same-time correlations only, so it does not capture lag structure.",
            "Strong calendar correlations can reflect seasonality rather than direct causal effects.",
        ],
    }
    figure_paths["interactive_correlation_heatmap"] = save_plotly_figure(plotly_fig, "correlation_heatmap_interactive.html", corr_meta)
    interactive_charts.append({"filename": "correlation_heatmap_interactive.html", "title": corr_meta["title"], "description": corr_meta["description"]})

    plotly_fig = px.line(lagged_corr, x="lag_hours", y="correlation", color="feature", title="Interactive Lagged Correlation Screen", labels={"lag_hours": "Lag (hours)", "correlation": "Correlation", "feature": "Feature"})
    lag_meta = {
        "title": "Lagged Correlations",
        "description": "Interactive lag scan showing how each exogenous variable correlates with price across prior hours.",
        "subtitle": "Built from the hourly aligned panel by shifting each feature from lag 0 to lag 48 hours and correlating it with price at time t.",
        "what_it_is": "A multi-line chart where each line shows the correlation between the target and a lagged version of one feature.",
        "data_used": "Columns used: the six exogenous variables in EXOGENOUS_COLUMNS. For each lag, the feature is shifted backward in time before correlation with price is computed.",
        "how_to_read": "The x-axis is the lag in hours. Lag 0 means same-time correlation. Higher lags mean using older feature values. The larger the absolute value on the y-axis, the stronger the linear association at that lag.",
        "what_it_shows": "Whether a variable has a stronger relationship with price immediately or after a historical delay, which helps identify candidate lag features for later forecasting models.",
        "why_it_matters": "This is more informative than a same-time correlation table because forecasting models usually rely on past values rather than contemporaneous unknown values.",
        "notes": [
            "This is still an exploratory screen, not a time-aware forecast evaluation.",
            "A strong lagged correlation does not guarantee out-of-sample predictive value at a 48-hour forecast horizon.",
        ],
    }
    figure_paths["interactive_lagged_correlations"] = save_plotly_figure(plotly_fig, "lagged_correlations_interactive.html", lag_meta)
    interactive_charts.append({"filename": "lagged_correlations_interactive.html", "title": lag_meta["title"], "description": lag_meta["description"]})

    missing_daily = hourly_panel[EXOGENOUS_COLUMNS].isna().sum(axis=1).resample("D").sum().rename("missing_values").reset_index()
    plotly_fig = px.line(missing_daily, x="time", y="missing_values", title="Interactive Missingness Over Time", labels={"time": "Time", "missing_values": "Missing value count"})
    missing_meta = {
        "title": "Missingness Over Time",
        "description": "Interactive daily view of how many exogenous values are missing in the hourly aligned panel.",
        "subtitle": "Uses the exogenous columns only and sums missing values by day after hourly alignment.",
        "what_it_is": "A time-series chart of daily missing-value counts across the exogenous features.",
        "data_used": "Columns used: temperature_2m, cloud_cover, wind_speed_10m, shortwave_radiation, total_load, generation_forecast. For each hourly row, missing values are counted, then aggregated by day.",
        "how_to_read": "Higher peaks mean more missing exogenous entries on that day. Zoom into spikes to locate missing blocks precisely and relate them to specific source issues.",
        "what_it_shows": "When missingness occurs, whether it is isolated or block-like, and whether the data problems are concentrated in a few periods.",
        "why_it_matters": "This helps decide whether later preprocessing should use imputation, exclusion, or explicit missingness indicators.",
        "notes": [
            "The largest spikes are driven mainly by missing generation_forecast blocks.",
            "The clean combined dataset hides these periods by dropping the affected timestamps entirely.",
        ],
    }
    figure_paths["interactive_missingness_over_time"] = save_plotly_figure(plotly_fig, "missingness_over_time_interactive.html", missing_meta)
    interactive_charts.append({"filename": "missingness_over_time_interactive.html", "title": missing_meta["title"], "description": missing_meta["description"]})

    scatter_features = ["total_load", "wind_speed_10m", "shortwave_radiation", "generation_forecast"]
    scatter_df = hourly_panel[[TARGET] + scatter_features].dropna()
    if len(scatter_df) > 12000:
        scatter_df = scatter_df.sample(12000, random_state=42)
    scatter_long = scatter_df.melt(id_vars=TARGET, var_name="feature", value_name="feature_value")
    plotly_fig = px.scatter(scatter_long, x="feature_value", y=TARGET, facet_col="feature", facet_col_wrap=2, opacity=0.2, title="Interactive Price vs Selected Exogenous Variables", labels={"feature_value": "Feature value", TARGET: "Price (EUR/MWh)"}, trendline="ols")
    plotly_fig.update_layout(height=850)
    scatter_meta = {
        "title": "Price vs Selected Exogenous Variables",
        "description": "Interactive faceted scatter view for the strongest exploratory exogenous candidates.",
        "subtitle": "Uses a sampled subset of the hourly aligned panel for readability and overlays OLS trend lines in each facet.",
        "what_it_is": "A set of scatter plots comparing price against selected exogenous variables in separate panels.",
        "data_used": "Columns used: price, total_load, wind_speed_10m, shortwave_radiation, generation_forecast from the hourly aligned panel. A random sample is used when the full set is too dense to display clearly.",
        "how_to_read": "Each dot is one timestamp. The x-axis is the feature value and the y-axis is price. The fitted line shows the average linear direction, while the point cloud shows how noisy the relationship is.",
        "what_it_shows": "Whether a variable appears to have a weak, moderate, or stronger linear relationship with price, and how much scatter remains around that trend.",
        "why_it_matters": "This helps separate variables with a visible directional signal from variables that are mostly noise or only useful after lagging or nonlinear modeling.",
        "notes": [
            "The plots use sampling for readability, so they are illustrative rather than exhaustive point-for-point dumps of the full dataset.",
            "The OLS trend line is exploratory and should not be treated as a forecasting model.",
        ],
    }
    figure_paths["interactive_scatter_grid"] = save_plotly_figure(plotly_fig, "price_vs_selected_exogenous_interactive.html", scatter_meta)
    interactive_charts.append({"filename": "price_vs_selected_exogenous_interactive.html", "title": scatter_meta["title"], "description": scatter_meta["description"]})

    weather_pairwise_metrics, weather_r2_summary = run_weather_pairwise_screen(hourly_panel.reset_index())
    weather_pairwise_metrics.to_csv(TABLE_DIR / "weather_pairwise_ols_metrics.csv", index=False)
    weather_r2_summary.to_csv(TABLE_DIR / "weather_r2_price_load.csv", index=False)
    figure_paths["weather_r2_vs_price"] = plot_weather_r2_bars(weather_r2_summary, TARGET, "weather_r2_vs_price.png")
    figure_paths["weather_r2_vs_load"] = plot_weather_r2_bars(weather_r2_summary, "total_load", "weather_r2_vs_load.png")
    figure_paths["weather_r2_comparison"] = plot_weather_r2_comparison(weather_r2_summary, "weather_r2_price_load_comparison.png")

    plotly_fig = px.bar(
        weather_r2_summary,
        x="feature",
        y=["r2_price", "r2_total_load"],
        barmode="group",
        title="Interactive Weather Pairwise OLS R² Comparison",
        labels={"value": "R²", "feature": "Weather feature", "variable": "Target"},
    )
    plotly_fig.update_traces(texttemplate="%{y:.3f}", textposition="outside")
    weather_r2_meta = {
        "title": "Weather R²: Price vs Total Load",
        "description": "Interactive grouped-bar comparison of pairwise weather OLS R² against price and total load.",
        "subtitle": "Each bar comes from a separate one-predictor OLS model fit on aligned non-null hourly rows only.",
        "what_it_is": "A grouped bar chart comparing how much each weather variable explains price versus total load under simple linear regression.",
        "data_used": "Columns used: temperature_2m, cloud_cover, wind_speed_10m, shortwave_radiation, price, and total_load from the hourly aligned panel.",
        "how_to_read": "For each weather feature, compare the two bars. A taller bar means that predictor explains more same-time variance in that target under a one-variable linear model.",
        "what_it_shows": "Whether a weather variable is more directly tied to load than to price, and which weather features are strongest in simple linear screening.",
        "why_it_matters": "This directly answers the teammate request and helps separate explanatory weather signals from features that may only become useful after lags or richer modeling.",
        "notes": [
            "These are pairwise linear fits, not forecast scores.",
            "Low R² does not imply that a weather feature is useless in a multivariate time-series model.",
        ],
    }
    figure_paths["interactive_weather_r2_comparison"] = save_plotly_figure(plotly_fig, "weather_r2_price_load_comparison_interactive.html", weather_r2_meta)
    interactive_charts.append({"filename": "weather_r2_price_load_comparison_interactive.html", "title": weather_r2_meta["title"], "description": weather_r2_meta["description"]})

    regression_metric_rows = []
    coefficient_tables = []

    for feature in EXOGENOUS_COLUMNS:
        result, fitted_df, coef_df = fit_ols_model(hourly_panel.reset_index(), TARGET, [feature], f"Simple OLS: price ~ {feature}", f"simple_{feature}")
        regression_metric_rows.append(build_regression_metric_row(result, "simple", feature))
        coefficient_tables.append(coef_df)
        plot_simple_regression(fitted_df, feature, TARGET, "Price (EUR/MWh)", result, f"regression_{feature}_vs_price.png")
        plot_regression_diagnostics(fitted_df, TARGET, "Price (EUR/MWh)", result, f"Diagnostics: price ~ {feature}", f"regression_diagnostics_{feature}.png")

    for feature_a, feature_b in combinations(EXOGENOUS_COLUMNS, 2):
        features = [feature_a, feature_b]
        label = feature_combo_label(features)
        slug = feature_combo_slug(features)
        result, fitted_df, coef_df = fit_ols_model(hourly_panel.reset_index(), TARGET, features, f"Pairwise OLS: price ~ {label}", f"pair_{slug}")
        regression_metric_rows.append(build_regression_metric_row(result, "pairwise", label))
        coefficient_tables.append(coef_df)
        plot_regression_diagnostics(fitted_df, TARGET, "Price (EUR/MWh)", result, f"Diagnostics: price ~ {label}", f"pair_regression_diagnostics_{slug}.png")

    for name, features, prefix in [
        ("Multiple OLS A: price ~ weather + production", EXOGENOUS_COLUMNS, "multiple_model_a"),
        ("Multiple OLS B: price ~ weather + production + calendar", EXOGENOUS_COLUMNS + CALENDAR_COLUMNS, "multiple_model_b"),
    ]:
        result, fitted_df, coef_df = fit_ols_model(hourly_panel.reset_index(), TARGET, features, name, prefix)
        regression_metric_rows.append(build_regression_metric_row(result, "multiple", ", ".join(features)))
        coefficient_tables.append(coef_df)
        diag_path = plot_regression_diagnostics(fitted_df, TARGET, "Price (EUR/MWh)", result, f"Diagnostics: {name}", f"{prefix}_diagnostics.png")
        if prefix == "multiple_model_b":
            figure_paths["regression_multi_b_diagnostics"] = diag_path
            figure_paths["regression_multi_b_coefficients"] = plot_standardized_coefficients(coef_df, features, fitted_df, "multiple_model_b_standardized_coefficients.png")
            interactive_df = fitted_df.copy()
            if len(interactive_df) > 15000:
                interactive_df = interactive_df.sample(15000, random_state=42)
            plotly_fig = px.scatter(interactive_df, x="predicted_value", y=TARGET, opacity=0.2, title="Interactive Actual vs Predicted Price (Multiple OLS B)", labels={"predicted_value": "Predicted price", TARGET: "Actual price"}, trendline="ols")
            regression_meta = {
                "title": "Regression Actual vs Predicted",
                "description": "Interactive diagnostic view for the strongest exploratory OLS model in the EDA.",
                "subtitle": f"Uses the multiple OLS B model with R²={result.r2:.3f}, Adj. R²={result.adj_r2:.3f}, explained variance={result.explained_variance:.3f}, model p-value={result.model_pvalue:.3g}.",
                "what_it_is": "A scatter plot comparing actual observed price against the model's fitted price for each row used by the regression.",
                "data_used": "Columns used: price, temperature_2m, cloud_cover, wind_speed_10m, shortwave_radiation, total_load, generation_forecast, hour_sin, hour_cos, day_of_week, month, is_weekend. Rows with missing values in any of these columns are dropped for this model.",
                "how_to_read": "If predictions were perfect, points would line up on a 45-degree diagonal. The wider the spread away from that direction, the more unexplained variation remains.",
                "what_it_shows": "How much of the target variation the linear model captures and where it misses badly, especially in high-volatility or extreme-price periods.",
                "why_it_matters": "This is a fast visual check that the model has some signal but still leaves substantial error, which supports using richer forecasting models later.",
                "notes": [
                    "This is an in-sample diagnostic for exploratory regression, not a future-horizon forecast evaluation.",
                    "The model's adjusted R-squared is about 0.150, so large residual structure is expected.",
                ],
            }
            figure_paths["interactive_regression_actual_vs_predicted"] = save_plotly_figure(plotly_fig, "multiple_model_b_actual_vs_predicted_interactive.html", regression_meta)
            interactive_charts.append({"filename": "multiple_model_b_actual_vs_predicted_interactive.html", "title": regression_meta["title"], "description": regression_meta["description"]})

    regression_metrics = pd.DataFrame(regression_metric_rows).sort_values(["model_type", "adj_r2"], ascending=[True, False])
    regression_metrics.to_csv(TABLE_DIR / "regression_metrics.csv", index=False)
    coefficient_table_all = pd.concat(coefficient_tables, ignore_index=True)
    coefficient_table_all.to_csv(TABLE_DIR / "regression_coefficients_all_models.csv", index=False)
    single_price_metrics = regression_metrics[(regression_metrics["target_column"] == TARGET) & (regression_metrics["model_type"] == "simple")].sort_values("adj_r2", ascending=False)
    pairwise_price_metrics = regression_metrics[(regression_metrics["target_column"] == TARGET) & (regression_metrics["model_type"] == "pairwise")].sort_values("adj_r2", ascending=False)
    single_price_interpretations, pairwise_price_interpretations = build_price_regression_interpretations(regression_metrics, coefficient_table_all)
    single_price_metrics.to_csv(TABLE_DIR / "price_single_variable_regression_metrics.csv", index=False)
    pairwise_price_metrics.to_csv(TABLE_DIR / "price_two_variable_regression_metrics.csv", index=False)
    single_price_interpretations.to_csv(TABLE_DIR / "price_single_variable_interpretations.csv", index=False)
    pairwise_price_interpretations.to_csv(TABLE_DIR / "price_two_variable_interpretations.csv", index=False)
    coefficient_table_all[coefficient_table_all["model"].str.startswith("Simple OLS: price ~")].to_csv(TABLE_DIR / "price_single_variable_regression_coefficients.csv", index=False)
    coefficient_table_all[coefficient_table_all["model"].str.startswith("Pairwise OLS: price ~")].to_csv(TABLE_DIR / "price_two_variable_regression_coefficients.csv", index=False)
    figure_paths["price_single_variable_regression_ranking"] = plot_regression_model_ranking(single_price_metrics, "price_single_variable_regression_r2_ranking.png", "Single-variable Price Regression Ranking", top_n=None, sort_col="adj_r2")
    figure_paths["price_pairwise_regression_ranking"] = plot_regression_model_ranking(pairwise_price_metrics, "price_two_variable_regression_r2_ranking.png", "Two-variable Price Regression Ranking", top_n=12, sort_col="adj_r2")

    promising_features = choose_promising_features(signal_screen, regression_metrics)
    recommendations = recommendation_paragraphs(promising_features, signal_screen, quality_table)
    lag_summary = (
        lagged_corr.assign(abs_correlation=lambda x: x["correlation"].abs())
        .sort_values(["feature", "abs_correlation"], ascending=[True, False])
        .groupby("feature")
        .head(1)
        .sort_values("abs_correlation", ascending=False)
    )
    lag_summary.to_csv(TABLE_DIR / "best_lagged_correlations.csv", index=False)

    recommended_figures = pd.DataFrame(
        [
            {"figure": "price_full_series.png", "what_it_shows": "Long-run target history with volatility shifts and extreme spikes.", "why_it_matters": "Establishes that the target is non-stationary and heavy-tailed."},
            {"figure": "price_frequency_transition.png", "what_it_shows": "The switch from hourly to quarter-hour price resolution at the start of 2025.", "why_it_matters": "Explains why the current dataset cannot yet support a fully aligned 15-minute exogenous panel."},
            {"figure": "missingness_over_time.png", "what_it_shows": "When exogenous missing blocks occur over time.", "why_it_matters": "Supports explicit imputation or exclusion decisions instead of silent row dropping."},
            {"figure": "correlation_heatmap_weather_production_price.png", "what_it_shows": "Initial structure among price, weather, system, and calendar variables.", "why_it_matters": "Quickly highlights candidate features and redundant relationships."},
            {"figure": "lagged_correlation_screen.png", "what_it_shows": "How exogenous-price correlation changes across historical lags.", "why_it_matters": "Guides later lag-feature engineering without overclaiming forecasting value."},
            {"figure": "weather_r2_price_load_comparison.png", "what_it_shows": "How strongly each weather variable explains price versus total load in pairwise OLS screening.", "why_it_matters": "Directly answers whether weather is more linearly tied to demand than to price."},
            {"figure": "price_two_variable_regression_r2_ranking.png", "what_it_shows": "Which two-variable price regressions explain the most variance among all exogenous feature pairs.", "why_it_matters": "Makes pair combinations like `cloud_cover + wind_speed_10m` comparable instead of anecdotal."},
            {"figure": "multiple_model_b_diagnostics.png", "what_it_shows": "Fit quality and residual behavior for the strongest exploratory linear model.", "why_it_matters": "Shows both the usable signal and the remaining modeling gap."},
            {"figure": "multiple_model_b_standardized_coefficients.png", "what_it_shows": "Relative influence of the multi-feature predictors on a comparable scale.", "why_it_matters": "Supports feature selection and interpretation discussions."},
            {"figure": "anomaly_candidates_price.png", "what_it_shows": "Flagged price spikes and unusual low-price periods.", "why_it_matters": "Highlights where robust forecasting methods and careful evaluation will matter most."},
        ]
    )
    recommended_figures.to_csv(TABLE_DIR / "recommended_figures.csv", index=False)
    figure_paths["interactive_index"] = write_interactive_index(interactive_charts)

    summary_payload = {
        "inventory": inventory_df.to_dict(orient="records"),
        "inventory_notes": notes_df.to_dict(orient="records"),
        "price_frequency_transition": price_transition,
        "gap_events": gap_table.to_dict(orient="records"),
        "quality_checks": quality_table.to_dict(orient="records"),
        "suspicious_value_checks": suspicious_table.to_dict(orient="records"),
        "signal_screen_top": signal_screen.head(10).to_dict(orient="records"),
        "weather_pairwise_metrics": weather_pairwise_metrics.to_dict(orient="records"),
        "weather_r2_summary": weather_r2_summary.to_dict(orient="records"),
        "price_single_variable_regressions": single_price_metrics.to_dict(orient="records"),
        "price_single_variable_interpretations": single_price_interpretations.to_dict(orient="records"),
        "price_two_variable_regressions": pairwise_price_metrics.to_dict(orient="records"),
        "price_two_variable_interpretations": pairwise_price_interpretations.to_dict(orient="records"),
        "regression_metrics": regression_metrics.to_dict(orient="records"),
        "leakage_risks": leakage_table.to_dict(orient="records"),
        "recommended_figures": recommended_figures.to_dict(orient="records"),
    }
    write_json(EDA_DIR / "data_summary.json", summary_payload)

    data_summary_md = f"""# Dataset Summary

## Inventory

{markdown_table(inventory_df.assign(date_start=inventory_df['date_start'].astype(str), date_end=inventory_df['date_end'].astype(str)))}

## Notes

{markdown_table(notes_df)}

## Main Alignment Findings

- Hourly aligned panel rows: {len(hourly_panel)}
- 2025 quarter-hour target rows: {len(price_2025_15m)}
- First quarter-hour price timestamp: {price_transition['first_non_hourly_timestamp']}
- Rows removed by `final_dataset_full_clean.csv` relative to `final_dataset_full_raw.csv`: {len(set(raw_join['time']) - set(clean_join['time']))}

## Strongest Initial Candidate Features

{markdown_table(signal_screen.head(6))}

## Best Price Regression Screens

### Single-variable

{markdown_table(single_price_metrics[["primary_feature", "r2", "adj_r2", "explained_variance"]].head(6))}

### Two-variable

{markdown_table(pairwise_price_metrics[["primary_feature", "r2", "adj_r2", "explained_variance"]].head(8))}
"""
    (EDA_DIR / "data_summary.md").write_text(data_summary_md, encoding="utf-8")

    key_numbers = {
        "hourly_only_segment_end": price_transition["hourly_only_segment_end"],
        "last_hourly_price_ts": price_transition["last_top_of_hour_timestamp"],
        "first_quarter_hour_ts": price_transition["first_non_hourly_timestamp"],
        "negative_price_count": int((hourly_panel[TARGET] < 0).sum()),
        "price_mean": float(hourly_panel[TARGET].mean()),
        "price_std": float(hourly_panel[TARGET].std()),
        "price_min": float(hourly_panel[TARGET].min()),
        "price_max": float(hourly_panel[TARGET].max()),
    }

    write_report(
        inventory_df=inventory_df,
        notes_df=notes_df,
        quality_table=quality_table,
        suspicious_table=suspicious_table,
        signal_screen=signal_screen,
        regression_metrics=regression_metrics,
        weather_pairwise_metrics=weather_pairwise_metrics,
        weather_r2_summary=weather_r2_summary,
        single_price_interpretations=single_price_interpretations,
        pairwise_price_interpretations=pairwise_price_interpretations,
        leakage_table=leakage_table,
        audit_summary=audit_summary,
        removed_times_df=removed_times_df,
        recommended_figures=recommended_figures,
        figure_paths=figure_paths,
        key_numbers=key_numbers,
        recommendations=recommendations,
    )
    write_findings_summary(recommendations, key_numbers, weather_r2_summary, regression_metrics, single_price_interpretations)
    write_technical_notes()
    write_documentation_addendum(weather_r2_summary, regression_metrics)
    write_regression_interpretation_notes(single_price_interpretations, pairwise_price_interpretations)

    print("EDA completed.")
    print(f"Report: {EDA_DIR / 'eda_report.md'}")
    print(f"Summary: {EDA_DIR / 'data_summary.md'}")
    print(f"Figures: {FIG_STATIC_DIR}")
    print(f"Interactive figures: {FIG_INTERACTIVE_DIR}")
    print(f"Tables: {TABLE_DIR}")
    print(f"Regression outputs: {REGRESSION_DIR}")
