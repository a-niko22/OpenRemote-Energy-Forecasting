from __future__ import annotations

from pathlib import Path
import json
import math


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
ASSETS_DIR = REPORTS_DIR / "assets"
PROPHET_JSON = ROOT / "benchmarks" / "prophet.json"

OUT_FIG = ASSETS_DIR / "prophet_vs_repo_comparison.svg"
OUT_MD = REPORTS_DIR / "prophet_benchmark_comparison.md"
OUT_HTML = REPORTS_DIR / "prophet_benchmark_comparison_dashboard.html"


def load_exp1_rows() -> list[dict]:
    return [
        {"group": "exp1a", "model": "exp1a_cnn_bilstm", "preprocess": "norm", "MAE": 29.6499, "RMSE": 41.2486, "MAPE": 4114.6316},
        {"group": "exp1a", "model": "exp1a_cnn_bilstm", "preprocess": "wavelet", "MAE": 28.2966, "RMSE": 39.3126, "MAPE": 3141.0965},
        {"group": "exp1a", "model": "exp1a_cnn_bilstm", "preprocess": "patch", "MAE": 31.2233, "RMSE": 42.6750, "MAPE": 4408.1151},
        {"group": "exp1b", "model": "exp1b_cnn_xlstm", "preprocess": "norm", "MAE": 26.5345, "RMSE": 39.3580, "MAPE": 4172.5943},
        {"group": "exp1b", "model": "exp1b_cnn_xlstm", "preprocess": "wavelet", "MAE": 27.1211, "RMSE": 39.3053, "MAPE": 3918.8283},
        {"group": "exp1b", "model": "exp1b_cnn_xlstm", "preprocess": "patch", "MAE": 28.8341, "RMSE": 40.6153, "MAPE": 4161.9200},
        {"group": "exp1c", "model": "exp1c_cnn_bilstm_transformer", "preprocess": "norm", "MAE": 37.6998, "RMSE": 50.8294, "MAPE": 4801.9512},
        {"group": "exp1c", "model": "exp1c_cnn_bilstm_transformer", "preprocess": "patch", "MAE": 32.2557, "RMSE": 44.9128, "MAPE": 4724.6062},
        {"group": "exp1c", "model": "exp1c_cnn_bilstm_transformer", "preprocess": "wavelet", "MAE": 39.8661, "RMSE": 53.1521, "MAPE": 4896.0546},
        {"group": "exp1d", "model": "exp1d_cnn_transformer", "preprocess": "norm", "MAE": 26.5262, "RMSE": 38.2869, "MAPE": 3093.2507},
        {"group": "exp1d", "model": "exp1d_cnn_transformer", "preprocess": "patch", "MAE": 29.6668, "RMSE": 40.9629, "MAPE": 3596.5461},
        {"group": "exp1d", "model": "exp1d_cnn_transformer", "preprocess": "wavelet", "MAE": 27.7646, "RMSE": 39.3107, "MAPE": 3560.1780},
    ]


def load_prophet_rows() -> list[dict]:
    payload = json.loads(PROPHET_JSON.read_text(encoding="utf-8"))
    held_out = payload["held_out_test"]
    cv = payload["cross_validation"]
    return [
        {
            "group": "prophet",
            "model": "prophet",
            "preprocess": "held_out_48h",
            "MAE": float(held_out["MAE"]),
            "RMSE": float(held_out["RMSE"]),
            "MAPE": float(held_out["MAPE"]),
            "coverage_pct": float(held_out["coverage_pct"]),
        },
        {
            "group": "prophet",
            "model": "prophet",
            "preprocess": "cross_validation",
            "MAE": float(cv["MAE"]),
            "RMSE": float(cv["RMSE"]),
            "MAPE": float(cv["MAPE"]),
            "coverage_pct": float(cv["coverage_pct"]),
        },
    ]


def build_rows() -> list[dict]:
    rows = load_exp1_rows() + load_prophet_rows()
    for row in rows:
        row["run"] = f"{row['model']}|{row['preprocess']}"
    return rows


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def draw_panel(
    lines: list[str],
    x0: int,
    y0: int,
    width: int,
    height: int,
    title: str,
    rows: list[dict],
    mode: str,
) -> None:
    lines.append(f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="12" fill="#fffdf8" stroke="#d8d0c4"/>')
    lines.append(f'<text x="{x0 + 14}" y="{y0 + 26}" font-size="18" font-weight="700" fill="#18201d">{escape_xml(title)}</text>')

    top = y0 + 48
    label_w = 150
    bar_w = width - label_w - 40
    row_h = (height - 66) / max(len(rows), 1)

    maes = [float(r["MAE"]) for r in rows]
    rmses = [float(r["RMSE"]) for r in rows]

    def scale(v: float, values: list[float]) -> float:
        if mode == "log":
            logs = [math.log10(max(x, 1e-12)) for x in values]
            lo = min(logs)
            hi = max(logs)
            if hi == lo:
                return 1.0
            return (math.log10(max(v, 1e-12)) - lo) / (hi - lo)
        lo = min(values)
        hi = max(values)
        if hi == lo:
            return 1.0
        return (v - lo) / (hi - lo)

    for i, row in enumerate(rows):
        y = top + i * row_h
        label = str(row["label"])
        mae = float(row["MAE"])
        rmse = float(row["RMSE"])

        mae_len = 8 + (bar_w - 8) * scale(mae, maes)
        rmse_len = 8 + (bar_w - 8) * scale(rmse, rmses)

        lines.append(f'<text x="{x0 + 14}" y="{y + 15:.1f}" font-size="11" fill="#33403b">{escape_xml(label)}</text>')

        bar_x = x0 + label_w
        lines.append(f'<rect x="{bar_x}" y="{y + 3:.1f}" width="{mae_len:.1f}" height="8" rx="4" fill="#1f77b4"/>')
        lines.append(f'<rect x="{bar_x}" y="{y + 14:.1f}" width="{rmse_len:.1f}" height="8" rx="4" fill="#ff7f0e"/>')

    legend_y = y0 + height - 12
    lines.append(f'<rect x="{x0 + 16}" y="{legend_y - 9}" width="12" height="8" fill="#1f77b4"/>')
    lines.append(f'<text x="{x0 + 34}" y="{legend_y - 2}" font-size="11" fill="#33403b">MAE</text>')
    lines.append(f'<rect x="{x0 + 82}" y="{legend_y - 9}" width="12" height="8" fill="#ff7f0e"/>')
    lines.append(f'<text x="{x0 + 100}" y="{legend_y - 2}" font-size="11" fill="#33403b">RMSE</text>')

    scale_note = "log10 scale" if mode == "log" else "linear scale"
    lines.append(f'<text x="{x0 + width - 120}" y="{legend_y - 2}" font-size="11" fill="#64706b">{scale_note}</text>')


def make_svg(rows: list[dict]) -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    order = [
        "exp1a_cnn_bilstm|norm", "exp1a_cnn_bilstm|wavelet", "exp1a_cnn_bilstm|patch",
        "exp1b_cnn_xlstm|norm", "exp1b_cnn_xlstm|wavelet", "exp1b_cnn_xlstm|patch",
        "exp1c_cnn_bilstm_transformer|norm", "exp1c_cnn_bilstm_transformer|patch", "exp1c_cnn_bilstm_transformer|wavelet",
        "exp1d_cnn_transformer|norm", "exp1d_cnn_transformer|patch", "exp1d_cnn_transformer|wavelet",
        "prophet|held_out_48h", "prophet|cross_validation",
    ]
    run_to_row = {r["run"]: r for r in rows}
    view = [run_to_row[k] for k in order if k in run_to_row]

    label_map = {
        "exp1a_cnn_bilstm|norm": "1a norm",
        "exp1a_cnn_bilstm|wavelet": "1a wavelet",
        "exp1a_cnn_bilstm|patch": "1a patch",
        "exp1b_cnn_xlstm|norm": "1b norm",
        "exp1b_cnn_xlstm|wavelet": "1b wavelet",
        "exp1b_cnn_xlstm|patch": "1b patch",
        "exp1c_cnn_bilstm_transformer|norm": "1c norm",
        "exp1c_cnn_bilstm_transformer|patch": "1c patch",
        "exp1c_cnn_bilstm_transformer|wavelet": "1c wavelet",
        "exp1d_cnn_transformer|norm": "1d norm",
        "exp1d_cnn_transformer|patch": "1d patch",
        "exp1d_cnn_transformer|wavelet": "1d wavelet",
        "prophet|held_out_48h": "prophet held-out",
        "prophet|cross_validation": "prophet cv",
    }

    all_rows = []
    for r in view:
        key = r["run"]
        all_rows.append({
            "label": label_map.get(key, key),
            "MAE": r["MAE"],
            "RMSE": r["RMSE"],
            "group": r["group"],
        })

    exp_rows = [r for r in all_rows if not str(r["label"]).startswith("prophet")]

    width = 1420
    height = 770
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f6f3ee"/>',
        '<text x="28" y="46" font-size="28" font-weight="800" fill="#18201d">Repository Experiments vs Prophet Benchmark</text>',
        '<text x="28" y="70" font-size="14" fill="#64706b">Left: all tracked runs on log scale. Right: Exp.1 runs on linear scale.</text>',
    ]

    draw_panel(lines, 28, 94, 668, 648, "All tracked runs", all_rows, mode="log")
    draw_panel(lines, 724, 94, 668, 648, "Exp.1 runs only", exp_rows, mode="linear")

    lines.append("</svg>")
    OUT_FIG.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_row(rows: list[dict], metric: str) -> dict:
    exp_only = [r for r in rows if r["group"] != "prophet"]
    return min(exp_only, key=lambda r: float(r[metric]))


def find_row(rows: list[dict], model: str, preprocess: str) -> dict:
    for r in rows:
        if r["model"] == model and r["preprocess"] == preprocess:
            return r
    raise ValueError(f"Missing row for {model}/{preprocess}")


def to_markdown(rows: list[dict]) -> str:
    best_mae = best_row(rows, "MAE")
    best_rmse = best_row(rows, "RMSE")

    prophet_hold = find_row(rows, "prophet", "held_out_48h")
    prophet_cv = find_row(rows, "prophet", "cross_validation")

    table = sorted(rows, key=lambda r: (r["model"], r["preprocess"]))

    lines = [
        "# Repository Experiments vs Prophet Benchmark",
        "",
        "This snapshot compares all currently tracked experiment metrics in this repository against the committed Prophet benchmark.",
        "",
        "## Comparison Figure",
        "",
        "![Repo experiments vs Prophet](./assets/prophet_vs_repo_comparison.svg)",
        "",
        "## Comparability Caveat",
        "",
        "- Exp.1 metrics are from electricity price snapshots: `reports/exp1_ab_results_snapshot.md` and `reports/exp1_cd_results_snapshot.md`.",
        "- Prophet metrics are from `benchmarks/prophet.json` generated on `prophet/mock_tariff.csv` (synthetic tariff data clipped to 0.01-0.50 EUR/kWh).",
        "- Because targets and scale differ, absolute MAE/RMSE values are not directly comparable across Exp.1 and Prophet.",
        "",
        "## Final Metrics Table",
        "",
        "| Model | Run | MAE | RMSE | MAPE | Coverage % |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for row in table:
        cov_val = row.get("coverage_pct")
        cov = "-" if cov_val is None else f"{float(cov_val):.2f}"
        lines.append(
            f"| `{row['model']}` | `{row['preprocess']}` | {float(row['MAE']):.6f} | {float(row['RMSE']):.6f} | {float(row['MAPE']):.4f} | {cov} |"
        )

    lines.extend([
        "",
        "## Quick Read",
        "",
        f"- Best Exp.1 MAE: `{best_mae['model']} + {best_mae['preprocess']}` ({float(best_mae['MAE']):.4f})",
        f"- Best Exp.1 RMSE: `{best_rmse['model']} + {best_rmse['preprocess']}` ({float(best_rmse['RMSE']):.4f})",
        f"- Prophet held-out (48h): MAE `{float(prophet_hold['MAE']):.6f}`, RMSE `{float(prophet_hold['RMSE']):.6f}`, coverage `{float(prophet_hold['coverage_pct']):.2f}%`",
        f"- Prophet cross-validation: MAE `{float(prophet_cv['MAE']):.6f}`, RMSE `{float(prophet_cv['RMSE']):.6f}`, coverage `{float(prophet_cv['coverage_pct']):.2f}%`",
        "",
        "## Interpretation",
        "",
        "- The visualization uses log scale for the all-run panel so both result ranges are visible.",
        "- The Exp.1-only panel keeps linear scale for fair ranking within existing deep-learning runs.",
        "- To benchmark Prophet fairly against Exp.1 models, rerun Prophet on the same target series and split settings.",
    ])

    return "\n".join(lines) + "\n"


def to_html(rows: list[dict]) -> str:
    order = [
        "exp1a_cnn_bilstm|norm", "exp1a_cnn_bilstm|wavelet", "exp1a_cnn_bilstm|patch",
        "exp1b_cnn_xlstm|norm", "exp1b_cnn_xlstm|wavelet", "exp1b_cnn_xlstm|patch",
        "exp1c_cnn_bilstm_transformer|norm", "exp1c_cnn_bilstm_transformer|patch", "exp1c_cnn_bilstm_transformer|wavelet",
        "exp1d_cnn_transformer|norm", "exp1d_cnn_transformer|patch", "exp1d_cnn_transformer|wavelet",
        "prophet|held_out_48h", "prophet|cross_validation",
    ]
    run_to_row = {r["run"]: r for r in rows}
    table = [run_to_row[k] for k in order if k in run_to_row]

    model_label = {
        "exp1a_cnn_bilstm": "Exp.1.a CNN-BiLSTM",
        "exp1b_cnn_xlstm": "Exp.1.b CNN-xLSTM",
        "exp1c_cnn_bilstm_transformer": "Exp.1.c CNN-BiLSTM-Transformer",
        "exp1d_cnn_transformer": "Exp.1.d CNN-Transformer",
        "prophet": "Prophet",
    }
    run_label = {
        "norm": "norm",
        "wavelet": "wavelet",
        "patch": "patch",
        "held_out_48h": "held_out_48h",
        "cross_validation": "cross_validation",
    }

    mae_values = [float(r["MAE"]) for r in table]
    rmse_values = [float(r["RMSE"]) for r in table]
    max_mae = max(mae_values) if mae_values else 1.0
    max_rmse = max(rmse_values) if rmse_values else 1.0

    table_rows = []
    mae_bars = []
    rmse_bars = []
    for row in table:
        cov_val = row.get("coverage_pct")
        cov = "-" if cov_val is None else f"{float(cov_val):.2f}"
        m = model_label.get(row["model"], row["model"])
        rp = run_label.get(row["preprocess"], row["preprocess"])
        kind_class = "prophet" if row["model"] == "prophet" else "exp"
        mae_w = max(4.0, (float(row["MAE"]) / max_mae) * 100.0)
        rmse_w = max(4.0, (float(row["RMSE"]) / max_rmse) * 100.0)

        table_rows.append(
            "<tr>"
            f"<td>{m}</td>"
            f"<td>{rp}</td>"
            f"<td class=\"num\">{float(row['MAE']):.6f}</td>"
            f"<td class=\"num\">{float(row['RMSE']):.6f}</td>"
            f"<td class=\"num\">{float(row['MAPE']):.4f}</td>"
            f"<td class=\"num\">{cov}</td>"
            "</tr>"
        )
        mae_bars.append(
            f"<div class=\"bar-row\"><span>{m} + {rp}</span><div class=\"bar-bg\"><div class=\"bar {kind_class}\" style=\"width: {mae_w:.2f}%\"></div></div><strong>{float(row['MAE']):.6f}</strong></div>"
        )
        rmse_bars.append(
            f"<div class=\"bar-row\"><span>{m} + {rp}</span><div class=\"bar-bg\"><div class=\"bar {kind_class}\" style=\"width: {rmse_w:.2f}%\"></div></div><strong>{float(row['RMSE']):.6f}</strong></div>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Repository Experiments vs Prophet Benchmark</title>
  <style>
    :root {{
      --bg: #f6f3ee;
      --ink: #18201d;
      --muted: #64706b;
      --line: #d8d0c4;
      --panel: #fffdf8;
      --accent: #0f766e;
      --accent-2: #b45309;
      --accent-3: #9f1239;
      --shadow: 0 18px 45px rgba(24, 32, 29, 0.08);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Aptos", "Segoe UI", Tahoma, sans-serif;
      line-height: 1.5;
    }}
    .wrap {{
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: linear-gradient(120deg, #fffdf8 0%, #edf4ef 54%, #f7eadb 100%);
      padding: 42px 0 30px;
    }}
    .eyebrow {{
      margin: 0 0 10px;
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(32px, 5vw, 54px);
      line-height: 1.04;
    }}
    .subtitle {{
      max-width: 920px;
      margin: 14px 0 0;
      color: var(--muted);
      font-size: 17px;
    }}
    main {{
      padding: 26px 0 56px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 18px;
      margin-top: 18px;
      box-shadow: var(--shadow);
    }}
    .stats {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }}
    .k {{
      font-size: 12px;
      color: var(--muted);
      letter-spacing: .05em;
      text-transform: uppercase;
      margin: 0 0 7px;
      font-weight: 700;
    }}
    .v {{
      margin: 0;
      font-size: 26px;
      font-weight: 800;
      line-height: 1.1;
    }}
    .n {{
      margin: 7px 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 290px 1fr 92px;
      gap: 12px;
      align-items: center;
      margin: 10px 0;
      font-size: 14px;
    }}
    .bar-bg {{
      height: 13px;
      border-radius: 999px;
      background: #e4ded4;
      overflow: hidden;
    }}
    .bar {{
      height: 100%;
      border-radius: inherit;
    }}
    .bar.exp {{
      background: var(--accent);
    }}
    .bar.prophet {{
      background: var(--accent-2);
    }}
    .chart-note {{
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .note {{
      border-left: 4px solid var(--accent-3);
      background: #fff8ec;
      padding: 14px 16px;
      color: #43362b;
      border-radius: 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      margin-top: 14px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-size: 14px;
    }}
    th {{
      background: #ebe5da;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .num {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    img {{
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    @media (max-width: 900px) {{
      .stats {{
        grid-template-columns: 1fr;
      }}
      .bar-row {{
        grid-template-columns: 1fr;
        gap: 6px;
      }}
      .num {{
        text-align: left;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <p class="eyebrow">Experiment Comparison</p>
      <h1>Repository Experiments vs Prophet Benchmark</h1>
      <p class="subtitle">
        Full comparison across all tracked Experiment 1 runs (1a/1b/1c/1d) and Prophet benchmark variants.
      </p>
    </div>
  </header>
  <main class="wrap">
    <section class="stats">
      <article class="panel">
        <p class="k">Total Compared Runs</p>
        <p class="v">{len(table)}</p>
        <p class="n">12 Exp.1 runs + 2 Prophet runs</p>
      </article>
      <article class="panel">
        <p class="k">Experiment Families</p>
        <p class="v">1a, 1b, 1c, 1d</p>
        <p class="n">Each with norm, wavelet, patch preprocessing</p>
      </article>
      <article class="panel">
        <p class="k">Prophet Variants</p>
        <p class="v">held-out + CV</p>
        <p class="n">Coverage is reported for both Prophet variants</p>
      </article>
    </section>

    <section class="panel">
      <h2 style="margin:0 0 10px;">All-Run Comparison Figure</h2>
      <img src="./assets/prophet_vs_repo_comparison.svg" alt="Repository experiments versus Prophet benchmark chart">
      <p class="chart-note">Left side uses log scale so Prophet and Exp.1 can appear together; right side keeps Exp.1-only linear ranking.</p>
    </section>

    <section class="panel">
      <h2 style="margin:0 0 10px;">MAE Comparison Across All Runs</h2>
      {''.join(mae_bars)}
      <p class="chart-note">Bar width is relative within this panel; lower MAE is better.</p>
    </section>

    <section class="panel">
      <h2 style="margin:0 0 10px;">RMSE Comparison Across All Runs</h2>
      {''.join(rmse_bars)}
      <p class="chart-note">Bar width is relative within this panel; lower RMSE is better.</p>
    </section>

    <section class="panel">
      <div class="note">
        Prophet metrics are computed on synthetic `prophet/mock_tariff.csv` while Exp.1 metrics come from electricity-price snapshots.
        Use this as a repository-level status comparison, not a strict apples-to-apples benchmark.
      </div>
    </section>

    <section class="panel">
      <h2 style="margin:0 0 10px;">Metrics Table</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Run</th>
            <th class="num">MAE</th>
            <th class="num">RMSE</th>
            <th class="num">MAPE</th>
            <th class="num">Coverage %</th>
          </tr>
        </thead>
        <tbody>
          {''.join(table_rows)}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    rows = build_rows()
    make_svg(rows)
    OUT_MD.write_text(to_markdown(rows), encoding="utf-8")
    OUT_HTML.write_text(to_html(rows), encoding="utf-8")
    print(f"Wrote {OUT_FIG}")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
