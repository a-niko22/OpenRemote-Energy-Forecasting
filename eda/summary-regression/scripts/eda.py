from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from eda.pipeline import run_eda
from eda.regression import (
    RegressionResult,
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
    ensure_directories,
    load_csv,
    markdown_table,
    save_plotly_figure,
    select_complete_periods,
    summarize_dataset,
    write_interactive_index,
    write_json,
)

__all__ = [
    "RegressionResult",
    "SOURCE_FILES",
    "REFERENCE_FILES",
    "EDA_DIR",
    "FIG_STATIC_DIR",
    "FIG_INTERACTIVE_DIR",
    "TABLE_DIR",
    "REGRESSION_DIR",
    "TARGET",
    "WEATHER_COLUMNS",
    "PRODUCTION_COLUMNS",
    "EXOGENOUS_COLUMNS",
    "CALENDAR_COLUMNS",
    "ensure_directories",
    "configure_plotting",
    "save_plotly_figure",
    "write_interactive_index",
    "write_json",
    "markdown_table",
    "load_csv",
    "add_calendar_features",
    "summarize_dataset",
    "select_complete_periods",
    "fit_ols_model",
    "compute_lagged_correlations",
    "build_dataset_inventory",
    "data_quality_checks",
    "summarize_frequency_transition",
    "run_weather_pairwise_screen",
    "plot_simple_regression",
    "plot_regression_diagnostics",
    "plot_standardized_coefficients",
    "plot_weather_r2_bars",
    "plot_weather_r2_comparison",
    "plot_regression_model_ranking",
    "build_regression_metric_row",
    "feature_combo_label",
    "feature_combo_slug",
    "build_price_regression_interpretations",
    "feature_signal_screen",
    "build_leakage_table",
    "audit_joined_datasets",
    "choose_promising_features",
    "recommendation_paragraphs",
    "write_report",
    "write_findings_summary",
    "write_technical_notes",
    "write_documentation_addendum",
    "write_regression_interpretation_notes",
    "run_eda",
]


def main() -> None:
    run_eda()


if __name__ == "__main__":
    main()
