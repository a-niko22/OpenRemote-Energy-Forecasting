from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.axes import Axes
from sklearn.metrics import explained_variance_score
from statsmodels.graphics.gofplots import qqplot

from eda.config import EDA_DIR, REGRESSION_DIR, TARGET, WEATHER_COLUMNS
from eda.utils import save_figure


@dataclass
class RegressionResult:
    model_name: str
    target_col: str
    feature_set: list[str]
    nobs: int
    dropped_rows: int
    r2: float
    adj_r2: float
    explained_variance: float
    model_pvalue: float
    aic: float
    bic: float
    summary_path: str


def fit_ols_model(
    df: pd.DataFrame,
    target_col: str,
    features: list[str],
    model_name: str,
    output_prefix: str,
) -> tuple[RegressionResult, pd.DataFrame, pd.DataFrame]:
    subset = df[[target_col] + features].dropna().copy()
    dropped = len(df) - len(subset)
    x = sm.add_constant(subset[features], has_constant="add")
    y = subset[target_col]
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x)
    explained = explained_variance_score(y, predictions)

    summary_path = REGRESSION_DIR / f"{output_prefix}_summary.txt"
    summary_path.write_text(model.summary().as_text(), encoding="utf-8")

    coef_table = pd.DataFrame(
        {
            "model": model_name,
            "variable": model.params.index,
            "coefficient": model.params.values,
            "p_value": model.pvalues.values,
            "ci_low": model.conf_int()[0].values,
            "ci_high": model.conf_int()[1].values,
        }
    )
    coef_table.to_csv(REGRESSION_DIR / f"{output_prefix}_coefficients.csv", index=False)

    fitted_df = subset.copy()
    fitted_df["predicted_value"] = predictions
    fitted_df["residual"] = y - predictions
    fitted_df.to_csv(REGRESSION_DIR / f"{output_prefix}_fitted_values.csv", index=False)

    result = RegressionResult(
        model_name=model_name,
        target_col=target_col,
        feature_set=features,
        nobs=int(model.nobs),
        dropped_rows=int(dropped),
        r2=float(model.rsquared),
        adj_r2=float(model.rsquared_adj),
        explained_variance=float(explained),
        model_pvalue=float(model.f_pvalue) if model.f_pvalue is not None else float("nan"),
        aic=float(model.aic),
        bic=float(model.bic),
        summary_path=str(summary_path.relative_to(EDA_DIR)).replace("\\", "/"),
    )
    return result, fitted_df, coef_table


def regression_metric_text(result: RegressionResult, include_adj: bool = True, include_pvalue: bool = True) -> str:
    lines = [f"R² = {result.r2:.3f}"]
    if include_adj:
        lines.append(f"Adj. R² = {result.adj_r2:.3f}")
    lines.append(f"Explained variance = {result.explained_variance:.3f}")
    if include_pvalue and not math.isnan(result.model_pvalue):
        lines.append(f"Model p-value = {result.model_pvalue:.3g}")
    return "\n".join(lines)


def annotate_metrics(ax: Axes, result: RegressionResult, include_adj: bool = True, include_pvalue: bool = True) -> None:
    text = regression_metric_text(result, include_adj=include_adj, include_pvalue=include_pvalue)
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
    )


def plot_simple_regression(
    fitted_df: pd.DataFrame,
    feature: str,
    target_col: str,
    target_label: str,
    result: RegressionResult,
    filename: str,
) -> str:
    plot_df = fitted_df[[feature, target_col, "predicted_value"]].dropna().copy()
    if len(plot_df) > 7000:
        plot_df = plot_df.sample(7000, random_state=42)
    plot_df = plot_df.sort_values(feature)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(data=plot_df, x=feature, y=target_col, alpha=0.18, s=16, ax=ax, color="#4c72b0")
    ax.plot(plot_df[feature], plot_df["predicted_value"], color="#c44e52", linewidth=2.0, label="OLS fitted line")
    ax.set_title(f"{feature} vs {target_label} with OLS Fit")
    ax.set_xlabel(feature)
    ax.set_ylabel(target_label)
    ax.legend()
    annotate_metrics(ax, result, include_adj=False, include_pvalue=True)
    return save_figure(fig, filename)


def plot_regression_diagnostics(
    fitted_df: pd.DataFrame,
    target_col: str,
    target_label: str,
    result: RegressionResult,
    title_prefix: str,
    filename: str,
) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].scatter(fitted_df["predicted_value"], fitted_df[target_col], alpha=0.18, s=14, color="#4c72b0")
    min_val = min(fitted_df["predicted_value"].min(), fitted_df[target_col].min())
    max_val = max(fitted_df["predicted_value"].max(), fitted_df[target_col].max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], color="#c44e52", linestyle="--")
    axes[0, 0].set_title("Actual vs Predicted")
    axes[0, 0].set_xlabel(f"Predicted {target_label.lower()}")
    axes[0, 0].set_ylabel(f"Actual {target_label.lower()}")

    axes[0, 1].scatter(fitted_df["predicted_value"], fitted_df["residual"], alpha=0.18, s=14, color="#55a868")
    axes[0, 1].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[0, 1].set_title("Residuals vs Predicted")
    axes[0, 1].set_xlabel(f"Predicted {target_label.lower()}")
    axes[0, 1].set_ylabel("Residual")

    sns.histplot(fitted_df["residual"], bins=40, kde=True, color="#8172b3", ax=axes[1, 0])
    axes[1, 0].set_title("Residual Histogram")
    axes[1, 0].set_xlabel("Residual")
    axes[1, 0].set_ylabel("Count")

    qqplot(fitted_df["residual"], line="s", ax=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot of Residuals")

    fig.text(
        0.98,
        0.97,
        regression_metric_text(result, include_adj=True, include_pvalue=True),
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
    )
    fig.suptitle(title_prefix, fontsize=16)
    fig.tight_layout(rect=(0, 0, 0.94, 0.95))
    return save_figure(fig, filename)


def plot_standardized_coefficients(coef_df: pd.DataFrame, features: list[str], fitted_df: pd.DataFrame, filename: str) -> str:
    y_std = fitted_df[TARGET].std(ddof=0)
    rows = []
    for feature in features:
        coef_row = coef_df[coef_df["variable"] == feature]
        if coef_row.empty:
            continue
        x_std = fitted_df[feature].std(ddof=0)
        scale = x_std / y_std if y_std else np.nan
        rows.append(
            {
                "variable": feature,
                "beta_std": coef_row["coefficient"].iloc[0] * scale,
                "ci_low": coef_row["ci_low"].iloc[0] * scale,
                "ci_high": coef_row["ci_high"].iloc[0] * scale,
            }
        )
    plot_df = pd.DataFrame(rows).sort_values("beta_std")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        plot_df["beta_std"],
        plot_df["variable"],
        xerr=[plot_df["beta_std"] - plot_df["ci_low"], plot_df["ci_high"] - plot_df["beta_std"]],
        fmt="o",
        color="#4c72b0",
        ecolor="#c44e52",
        capsize=4,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Standardized Coefficients for the Multi-feature Model")
    ax.set_xlabel("Standardized coefficient")
    ax.set_ylabel("Feature")
    return save_figure(fig, filename)


def build_regression_metric_row(result: RegressionResult, model_type: str, primary_feature: str) -> dict:
    return {
        "model_name": result.model_name,
        "target_column": result.target_col,
        "model_type": model_type,
        "primary_feature": primary_feature,
        "feature_count": len(result.feature_set),
        "feature_list": ", ".join(result.feature_set),
        "nobs": result.nobs,
        "dropped_rows": result.dropped_rows,
        "r2": round(result.r2, 4),
        "adj_r2": round(result.adj_r2, 4),
        "explained_variance": round(result.explained_variance, 4),
        "model_pvalue": result.model_pvalue,
        "aic": round(result.aic, 2),
        "bic": round(result.bic, 2),
        "summary_path": result.summary_path,
    }


def rank_weather_pairwise_models(weather_metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = weather_metrics.pivot(index="feature", columns="target_column")
    pivot.columns = [f"{metric}_{target}" for metric, target in pivot.columns]
    summary = pivot.reset_index()
    for col in ["r2_price", "r2_total_load", "adj_r2_price", "adj_r2_total_load", "explained_variance_price", "explained_variance_total_load"]:
        if col not in summary.columns:
            summary[col] = np.nan
    summary["delta_r2_load_minus_price"] = summary["r2_total_load"] - summary["r2_price"]

    def classify(row: pd.Series) -> str:
        price_r2 = row.get("r2_price", np.nan)
        load_r2 = row.get("r2_total_load", np.nan)
        if (pd.isna(price_r2) and pd.isna(load_r2)) or max(price_r2, load_r2) < 0.01:
            return "weak direct linear signal"
        if load_r2 - price_r2 > 0.005:
            return "better for load"
        if price_r2 - load_r2 > 0.005:
            return "better for direct price screening"
        return "similar explanatory power"

    summary["recommendation"] = summary.apply(classify, axis=1)
    return summary.sort_values("r2_total_load", ascending=False)


def plot_weather_r2_bars(weather_summary: pd.DataFrame, target_col: str, filename: str) -> str:
    target_label = "Price" if target_col == TARGET else "Total Load"
    r2_col = f"r2_{target_col}"
    p_col = f"model_pvalue_{target_col}"
    plot_df = weather_summary[["feature", r2_col, p_col]].dropna().sort_values(r2_col, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(plot_df["feature"], plot_df[r2_col], color="#4c72b0")
    ax.set_title(f"Pairwise Weather OLS R² vs {target_label}")
    ax.set_xlabel("Weather feature")
    ax.set_ylabel("R²")
    ax.set_ylim(0, max(0.05, plot_df[r2_col].max() * 1.2 if not plot_df.empty else 0.05))
    ax.tick_params(axis="x", rotation=20)
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"R²={row[r2_col]:.3f}\np={row[p_col]:.2g}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.text(
        0.01,
        -0.02,
        "Note: each bar is a separate simple OLS fit using one weather variable only. These are explanatory screening scores, not forecasting scores.",
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    return save_figure(fig, filename)


def plot_weather_r2_comparison(weather_summary: pd.DataFrame, filename: str) -> str:
    plot_df = weather_summary[["feature", "r2_price", "r2_total_load"]].copy()
    plot_df = plot_df.melt(id_vars="feature", var_name="target", value_name="r2")
    plot_df["target"] = plot_df["target"].map({"r2_price": "Price", "r2_total_load": "Total Load"})
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=plot_df, x="feature", y="r2", hue="target", ax=ax, palette=["#4c72b0", "#55a868"])
    ax.set_title("Pairwise Weather OLS R²: Price vs Total Load")
    ax.set_xlabel("Weather feature")
    ax.set_ylabel("R²")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Target")
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 0.001,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.text(
        0.01,
        -0.02,
        "Interpretation: higher bars mean the weather feature explains more same-time variance in the target under a one-predictor linear model.",
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    return save_figure(fig, filename)


def run_weather_pairwise_screen(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for target_col in [TARGET, "total_load"]:
        for feature in WEATHER_COLUMNS:
            prefix = f"weather_{target_col}_{feature}"
            model_name = f"Pairwise OLS: {target_col} ~ {feature}"
            result, _, coef_df = fit_ols_model(df, target_col, [feature], model_name, prefix)
            coef_row = coef_df[coef_df["variable"] == feature].iloc[0]
            rows.append(
                {
                    "target_column": target_col,
                    "feature": feature,
                    "model_name": result.model_name,
                    "nobs": result.nobs,
                    "dropped_rows": result.dropped_rows,
                    "r2": result.r2,
                    "adj_r2": result.adj_r2,
                    "explained_variance": result.explained_variance,
                    "model_pvalue": result.model_pvalue,
                    "coefficient": coef_row["coefficient"],
                    "coefficient_p_value": coef_row["p_value"],
                    "ci_low": coef_row["ci_low"],
                    "ci_high": coef_row["ci_high"],
                    "summary_path": result.summary_path,
                }
            )
    metrics_df = pd.DataFrame(rows).sort_values(["target_column", "r2"], ascending=[True, False])
    summary_df = rank_weather_pairwise_models(metrics_df)
    return metrics_df, summary_df


def feature_combo_label(features: list[str] | tuple[str, ...]) -> str:
    return " + ".join(features)


def feature_combo_slug(features: list[str] | tuple[str, ...]) -> str:
    return "__".join(features)


def explanatory_power_label(r2: float) -> str:
    if pd.isna(r2):
        return "unknown explanatory power"
    if r2 < 0.01:
        return "very weak explanatory power"
    if r2 < 0.05:
        return "weak explanatory power"
    if r2 < 0.15:
        return "moderate explanatory power"
    return "stronger explanatory power"


def direction_text(coefficient: float, feature: str, target_label: str = "price") -> str:
    if pd.isna(coefficient):
        return f"No directional interpretation available for {feature}."
    if coefficient > 0:
        return f"As {feature} rises, fitted same-time {target_label} also rises on average."
    if coefficient < 0:
        return f"As {feature} rises, fitted same-time {target_label} falls on average."
    return f"As {feature} changes, the fitted same-time {target_label} is effectively flat in this model."


def build_price_regression_interpretations(
    regression_metrics: pd.DataFrame,
    coefficient_table_all: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    single_models = regression_metrics[
        (regression_metrics["target_column"] == TARGET) & (regression_metrics["model_type"] == "simple")
    ].copy()
    pair_models = regression_metrics[
        (regression_metrics["target_column"] == TARGET) & (regression_metrics["model_type"] == "pairwise")
    ].copy()

    single_rows = []
    for _, row in single_models.iterrows():
        coef_row = coefficient_table_all[
            (coefficient_table_all["model"] == row["model_name"]) & (coefficient_table_all["variable"] != "const")
        ].iloc[0]
        significant = bool(float(row["model_pvalue"]) < 0.05) if not pd.isna(row["model_pvalue"]) else False
        if significant and float(row["r2"]) < 0.01:
            note = "Statistically significant, but the model explains almost none of the price variance."
        elif significant:
            note = "Statistically significant with some explanatory value, but still only an in-sample linear screen."
        else:
            note = "Not statistically significant in this simple linear screen."
        single_rows.append(
            {
                "predictor": row["primary_feature"],
                "coefficient": coef_row["coefficient"],
                "coefficient_p_value": coef_row["p_value"],
                "model_pvalue": row["model_pvalue"],
                "r2": row["r2"],
                "adj_r2": row["adj_r2"],
                "explained_variance": row["explained_variance"],
                "nobs": row["nobs"],
                "direction": "rises" if coef_row["coefficient"] > 0 else "falls" if coef_row["coefficient"] < 0 else "flat",
                "statistically_significant_0_05": significant,
                "explanatory_power": explanatory_power_label(float(row["r2"])),
                "plain_language": direction_text(float(coef_row["coefficient"]), row["primary_feature"]),
                "interpretation_note": note,
                "summary_path": row["summary_path"],
            }
        )

    pair_rows = []
    for _, row in pair_models.iterrows():
        model_coefs = coefficient_table_all[
            (coefficient_table_all["model"] == row["model_name"]) & (coefficient_table_all["variable"] != "const")
        ].copy()
        descriptors = []
        for _, coef_row in model_coefs.iterrows():
            move = "rises" if coef_row["coefficient"] > 0 else "falls" if coef_row["coefficient"] < 0 else "stays flat"
            sig = "significant" if coef_row["p_value"] < 0.05 else "not significant"
            descriptors.append(
                f"{coef_row['variable']}: price {move} when the variable rises ({sig}, coef={coef_row['coefficient']:.4f}, p={coef_row['p_value']:.3g})"
            )
        pair_rows.append(
            {
                "model_features": row["primary_feature"],
                "feature_list": row["feature_list"],
                "model_pvalue": row["model_pvalue"],
                "r2": row["r2"],
                "adj_r2": row["adj_r2"],
                "explained_variance": row["explained_variance"],
                "nobs": row["nobs"],
                "feature_effects": " | ".join(descriptors),
                "interpretation_note": f"Overall model is {'statistically significant' if row['model_pvalue'] < 0.05 else 'not statistically significant'} with {explanatory_power_label(float(row['r2']))}.",
                "summary_path": row["summary_path"],
            }
        )

    return (
        pd.DataFrame(single_rows).sort_values("adj_r2", ascending=False),
        pd.DataFrame(pair_rows).sort_values("adj_r2", ascending=False),
    )


def plot_regression_model_ranking(
    metrics_df: pd.DataFrame,
    filename: str,
    title: str,
    top_n: int | None = None,
    sort_col: str = "adj_r2",
) -> str:
    plot_df = metrics_df.sort_values(sort_col, ascending=False).copy()
    if top_n is not None:
        plot_df = plot_df.head(top_n)
    plot_df = plot_df.sort_values(sort_col, ascending=True)
    fig_height = max(4.5, 0.6 * len(plot_df) + 1.6)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    bars = ax.barh(plot_df["primary_feature"], plot_df[sort_col], color="#4c72b0")
    label = "Adjusted R²" if sort_col == "adj_r2" else "R²"
    ax.set_title(title)
    ax.set_xlabel(label)
    ax.set_ylabel("Model specification")
    xmax = plot_df[sort_col].max() if not plot_df.empty else 0.0
    ax.set_xlim(0, max(0.06, xmax * 1.45 if xmax else 0.06))
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        ax.text(
            bar.get_width() + max(0.001, xmax * 0.02 if xmax else 0.001),
            bar.get_y() + bar.get_height() / 2,
            f"R²={row['r2']:.3f} | Adj. R²={row['adj_r2']:.3f} | EV={row['explained_variance']:.3f} | p={row['model_pvalue']:.2g}",
            va="center",
            ha="left",
            fontsize=9,
        )
    fig.text(
        0.01,
        -0.02,
        "Each bar is one OLS specification against price. Metrics are in-sample explanatory scores and should not be interpreted as forecast performance.",
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    return save_figure(fig, filename)
