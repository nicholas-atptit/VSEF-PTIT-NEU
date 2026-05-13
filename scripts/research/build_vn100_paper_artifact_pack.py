"""Build paper-ready VN100 NCKH tables, figures, and notes from official artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "outputs" / "vn100_hybrid_official_2025_confidence_sweep_traincutoff"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "generated"
TABLE_DIR_NAME = "paper_tables"
FIGURE_DIR_NAME = "paper_figures"
NOTE_DIR_NAME = "paper_notes"


TABLE_META = {
    "table1_dataset_evaluation_scope": {
        "source": "run_config.json; manifest.json; usable_cache_summary.csv; daily/hourly benchmark_summary.json",
        "claim": "The official run is a 2025 held-out VN100 benchmark with limited usable-cache coverage.",
        "limitation": "Standalone daily cache rows are not benchmark-usable; the daily benchmark uses hybrid resampled inputs.",
        "status": "ready",
    },
    "table2_model_baseline_list": {
        "source": "run_config.json; daily/hourly baseline_summary.csv",
        "claim": "The study compares LightGBM, XGBoost, random forest, stacking, and simple directional baselines.",
        "limitation": "No new model families are introduced by this artifact pack.",
        "status": "ready",
    },
    "table3_global_benchmark_results": {
        "source": "daily/hourly benchmark_summary.json; daily/hourly classification_accuracy_summary.csv",
        "claim": "The official benchmark produced nonzero predictions but did not pass the global 60% threshold.",
        "limitation": "Single official 2025 window with seven evaluated tickers.",
        "status": "ready",
    },
    "table4_baseline_delta_summary": {
        "source": "daily/hourly baseline_delta_summary.csv",
        "claim": "Some model/horizon rows outperform simple directional baselines, but this is not a global pass.",
        "limitation": "Baseline deltas are directional-accuracy diagnostics, not cost-adjusted returns.",
        "status": "ready",
    },
    "table5_confidence_filtered_diagnostics": {
        "source": "confidence_threshold_sweep_summary.csv; daily/hourly confidence_filter_summary.csv",
        "claim": "The selected hourly stacking h=1 threshold 0.57 slice passes 60% only at about 31.30% coverage.",
        "limitation": "Daily threshold-sweep rows are missing; available sweep rows cover hourly stacking h=1.",
        "status": "partial",
    },
    "table6_regime_specific_diagnostics": {
        "source": "daily/hourly regime_accuracy_summary.csv; generated ticker concentration summary",
        "claim": "Regime diagnostics show conditional signal, especially daily bear-regime h=20, without proving global performance.",
        "limitation": "Regime findings are post-hoc diagnostics from one official window unless validated ex ante.",
        "status": "ready",
    },
    "table7_statistical_significance_summary": {
        "source": "daily/hourly significance_summary.csv; daily/hourly mcnemar_summary.csv",
        "claim": "Several model/horizon rows are statistically above a 50% null.",
        "limitation": "Significance alone does not establish trading readiness or multi-window stability.",
        "status": "partial",
    },
    "table8_robustness_limitation_matrix": {
        "source": "artifact verification, concentration, coverage, cost/slippage readiness, model/source health artifacts",
        "claim": "The main limitations are limited coverage, selected-slice concentration, partial sweep evidence, and missing trading-cost validation.",
        "limitation": "The matrix records current evidence gaps; it is not a new experiment.",
        "status": "ready",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VN100 NCKH paper artifact pack from official outputs.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[dict[str, Any]], *, max_rows: int | None = None) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    displayed = rows if max_rows is None else rows[:max_rows]
    for row in displayed:
        lines.append("| " + " | ".join(format_cell(row.get(header, "")) for header in headers) + " |")
    if max_rows is not None and len(rows) > max_rows:
        lines.append("| " + " | ".join(["..."] + ["" for _ in headers[1:]]) + " |")
    return "\n".join(lines)


def write_table_markdown(path: Path, title: str, rows: list[dict[str, Any]], headers: list[str], meta: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [
        f"# {title}",
        "",
        markdown_table(headers, rows),
        "",
        "## Note",
        "",
        f"- Source artifact: {meta['source'].rstrip('.')}.",
        f"- Claim supported: {meta['claim'].rstrip('.')}.",
        f"- Limitation: {meta['limitation'].rstrip('.')}.",
        f"- Status: {meta['status']}.",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int:
    parsed = to_float(value)
    return int(parsed) if parsed is not None else 0


def to_bool_text(value: Any) -> str:
    return "true" if str(value).strip().lower() in {"true", "1", "yes"} else "false"


def percent(value: Any) -> str:
    parsed = to_float(value)
    if parsed is None:
        return ""
    return f"{parsed * 100:.2f}%"


def sort_key(row: dict[str, Any], keys: list[str]) -> tuple[Any, ...]:
    values: list[Any] = []
    for key in keys:
        value = row.get(key, "")
        parsed = to_float(value)
        values.append(parsed if parsed is not None else str(value))
    return tuple(values)


def add_meta(row: dict[str, Any], table_key: str) -> dict[str, Any]:
    meta = TABLE_META[table_key]
    return {
        **row,
        "source_artifact": meta["source"],
        "claim_supported": meta["claim"],
        "limitation": meta["limitation"],
        "status": meta["status"],
    }


def usable_cache_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for row in rows:
        frequency = str(row.get("frequency", "")).strip()
        if not frequency:
            continue
        item = counts.setdefault(
            frequency,
            {"usable": 0, "total": 0, "tickers": set(), "actual_start": set(), "actual_end": set()},
        )
        item["total"] += 1
        if to_bool_text(row.get("benchmark_usable")) == "true":
            item["usable"] += 1
            item["tickers"].add(str(row.get("ticker", "")).strip())
            if row.get("actual_start"):
                item["actual_start"].add(row["actual_start"])
            if row.get("actual_end"):
                item["actual_end"].add(row["actual_end"])
    return counts


def build_table1(artifact_dir: Path) -> list[dict[str, Any]]:
    run_config = read_json(artifact_dir / "run_config.json")
    manifest = read_json(artifact_dir / "manifest.json")
    daily_summary = read_json(artifact_dir / "daily" / "benchmark_summary.json")
    hourly_summary = read_json(artifact_dir / "hourly" / "benchmark_summary.json")
    cache_counts = usable_cache_counts(read_csv(artifact_dir / "usable_cache_summary.csv"))
    evaluated_tickers = sorted(set(daily_summary.get("evaluated_tickers", [])) | set(hourly_summary.get("evaluated_tickers", [])))

    rows = [
        {"item": "Universe", "value": run_config.get("universe", "VN100"), "detail": "Official benchmark universe."},
        {
            "item": "Raw daily cache request range",
            "value": f"{run_config.get('daily_start')} to {run_config.get('daily_end')}",
            "detail": f"Manifest raw daily range: {manifest.get('raw_daily_range', {}).get('start')} to {manifest.get('raw_daily_range', {}).get('end')}.",
        },
        {
            "item": "Raw hourly cache request range",
            "value": f"{run_config.get('hourly_start')} to {run_config.get('hourly_end')}",
            "detail": f"Manifest raw hourly range: {manifest.get('raw_hourly_range', {}).get('start')} to {manifest.get('raw_hourly_range', {}).get('end')}.",
        },
        {
            "item": "Training-label cutoff",
            "value": run_config.get("train_cutoff", ""),
            "detail": f"Rule: {run_config.get('training_label_cutoff_rule', '')}.",
        },
        {
            "item": "Official evaluation window",
            "value": f"{run_config.get('eval_start')} to {run_config.get('eval_end')}",
            "detail": "Held-out 2025 target outcomes.",
        },
        {
            "item": "Effective daily evaluation range",
            "value": f"{daily_summary.get('effective_eval_start')} to {daily_summary.get('effective_eval_end')}",
            "detail": f"Daily predictions: {daily_summary.get('n_predictions')}.",
        },
        {
            "item": "Effective hourly evaluation range",
            "value": f"{hourly_summary.get('effective_eval_start')} to {hourly_summary.get('effective_eval_end')}",
            "detail": f"Hourly predictions: {hourly_summary.get('n_predictions')}.",
        },
        {
            "item": "Evaluated tickers",
            "value": ", ".join(evaluated_tickers),
            "detail": f"{len(evaluated_tickers)} tickers evaluated in official summaries.",
        },
    ]
    for frequency in ("daily", "hourly"):
        item = cache_counts.get(frequency, {"usable": 0, "total": 0, "tickers": set(), "actual_start": set(), "actual_end": set()})
        starts = sorted(item["actual_start"])
        ends = sorted(item["actual_end"])
        rows.append(
            {
                "item": f"{frequency.title()} benchmark-usable cache rows",
                "value": f"{item['usable']} of {item['total']}",
                "detail": f"Usable tickers: {', '.join(sorted(item['tickers'])) or 'none'}; actual range: {starts[0] if starts else 'n/a'} to {ends[-1] if ends else 'n/a'}.",
            }
        )
    return [add_meta(row, "table1_dataset_evaluation_scope") for row in rows]


def build_table2(artifact_dir: Path) -> list[dict[str, Any]]:
    run_config = read_json(artifact_dir / "run_config.json")
    rows: list[dict[str, Any]] = []
    for model in sorted(run_config.get("models", [])):
        rows.append(
            {
                "type": "model",
                "name": model,
                "frequency": "daily/hourly",
                "horizons": f"daily={run_config.get('daily_horizons', [])}; hourly={run_config.get('hourly_horizons', [])}",
                "role": "machine-learning classifier in official benchmark",
            }
        )
    baseline_names = set()
    for frequency in ("daily", "hourly"):
        for row in read_csv(artifact_dir / frequency / "baseline_summary.csv"):
            baseline_names.add(str(row.get("baseline", "")).strip())
    for baseline in sorted(baseline_names):
        rows.append(
            {
                "type": "baseline",
                "name": baseline,
                "frequency": "daily/hourly",
                "horizons": "same evaluated horizons where baseline rows exist",
                "role": "directional comparison baseline",
            }
        )
    return [add_meta(row, "table2_model_baseline_list") for row in rows]


def build_table3(artifact_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frequency in ("daily", "hourly"):
        summary = read_json(artifact_dir / frequency / "benchmark_summary.json")
        accuracy_rows = read_csv(artifact_dir / frequency / "classification_accuracy_summary.csv")
        best_accuracy_row = (
            sorted(accuracy_rows, key=lambda row: to_float(row.get("accuracy")) or -1.0, reverse=True)[0]
            if accuracy_rows
            else {}
        )
        rows.append(
            {
                "frequency": frequency,
                "overall_accuracy": summary.get("overall_accuracy", ""),
                "n_predictions": summary.get("n_predictions", ""),
                "best_model_accuracy": summary.get("best_model_accuracy", ""),
                "best_model": best_accuracy_row.get("model", ""),
                "best_model_horizon": summary.get("best_model_horizon", ""),
                "best_baseline_accuracy": summary.get("best_baseline_accuracy", ""),
                "best_model_delta_vs_best_baseline": summary.get("best_model_delta_vs_best_baseline", ""),
                "passed_60pct_global": to_bool_text(summary.get("passed")),
                "evaluated_tickers": ", ".join(summary.get("evaluated_tickers", [])),
            }
        )
    return [add_meta(row, "table3_global_benchmark_results") for row in rows]


def build_table4(artifact_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frequency in ("daily", "hourly"):
        for row in read_csv(artifact_dir / frequency / "baseline_delta_summary.csv"):
            rows.append(
                {
                    "frequency": frequency,
                    "model": row.get("model", ""),
                    "horizon": row.get("horizon", ""),
                    "baseline": row.get("baseline", ""),
                    "model_accuracy": row.get("model_accuracy", ""),
                    "baseline_accuracy": row.get("baseline_accuracy", ""),
                    "accuracy_delta": row.get("accuracy_delta", ""),
                    "model_better_than_baseline": to_bool_text(row.get("model_better_than_baseline")),
                    "model_n_obs": row.get("model_n_obs", ""),
                }
            )
    rows.sort(key=lambda row: (row["frequency"], row["model"], to_int(row["horizon"]), row["baseline"]))
    return [add_meta(row, "table4_baseline_delta_summary") for row in rows]


def build_table5(artifact_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sweep_rows = read_csv(artifact_dir / "confidence_threshold_sweep_summary.csv")
    if sweep_rows:
        eligible_floors = [0.5, 0.4, 0.3]
        for floor in eligible_floors:
            eligible = [row for row in sweep_rows if (to_float(row.get("coverage_ratio")) or -1.0) >= floor]
            if eligible:
                best = sorted(
                    eligible,
                    key=lambda row: (
                        -(to_float(row.get("filtered_accuracy")) or -1.0),
                        -(to_float(row.get("coverage_ratio")) or -1.0),
                    ),
                )[0]
                rows.append(
                    {
                        "scope": f"best_at_coverage_floor_{floor:.0%}",
                        "frequency": best.get("frequency", ""),
                        "model": best.get("model", ""),
                        "horizon": best.get("horizon", ""),
                        "threshold": best.get("threshold", ""),
                        "total_rows": best.get("total_rows", ""),
                        "evaluated_rows": best.get("evaluated_rows", ""),
                        "coverage_ratio": best.get("coverage_ratio", ""),
                        "filtered_accuracy": best.get("filtered_accuracy", ""),
                        "passed_60pct": to_bool_text(best.get("passed_60pct")),
                        "selected_candidate": to_bool_text(best.get("selected_candidate")),
                    }
                )
            else:
                rows.append(
                    {
                        "scope": f"best_at_coverage_floor_{floor:.0%}",
                        "frequency": "",
                        "model": "",
                        "horizon": "",
                        "threshold": "",
                        "total_rows": "",
                        "evaluated_rows": "",
                        "coverage_ratio": "",
                        "filtered_accuracy": "",
                        "passed_60pct": "false",
                        "selected_candidate": "false",
                    }
                )
        for row in sweep_rows:
            if to_bool_text(row.get("selected_candidate")) == "true":
                rows.append(
                    {
                        "scope": "selected_candidate",
                        "frequency": row.get("frequency", ""),
                        "model": row.get("model", ""),
                        "horizon": row.get("horizon", ""),
                        "threshold": row.get("threshold", ""),
                        "total_rows": row.get("total_rows", ""),
                        "evaluated_rows": row.get("evaluated_rows", ""),
                        "coverage_ratio": row.get("coverage_ratio", ""),
                        "filtered_accuracy": row.get("filtered_accuracy", ""),
                        "passed_60pct": to_bool_text(row.get("passed_60pct")),
                        "selected_candidate": "true",
                    }
                )
    for frequency in ("daily", "hourly"):
        filter_rows = read_csv(artifact_dir / frequency / "confidence_filter_summary.csv")
        if not filter_rows:
            rows.append(
                {
                    "scope": f"{frequency}_confidence_filter_summary",
                    "frequency": frequency,
                    "model": "",
                    "horizon": "",
                    "threshold": "",
                    "total_rows": "",
                    "evaluated_rows": "",
                    "coverage_ratio": "",
                    "filtered_accuracy": "",
                    "passed_60pct": "false",
                    "selected_candidate": "false",
                }
            )
            continue
        best_filter = sorted(
            filter_rows,
            key=lambda row: (
                -(to_float(row.get("filtered_accuracy")) or -1.0),
                -(to_float(row.get("coverage_ratio")) or -1.0),
            ),
        )[0]
        rows.append(
            {
                "scope": f"best_default_filter_{frequency}",
                "frequency": frequency,
                "model": best_filter.get("model", ""),
                "horizon": best_filter.get("horizon", ""),
                "threshold": best_filter.get("confidence_threshold", ""),
                "total_rows": best_filter.get("total_rows", ""),
                "evaluated_rows": best_filter.get("evaluated_rows", ""),
                "coverage_ratio": best_filter.get("coverage_ratio", ""),
                "filtered_accuracy": best_filter.get("filtered_accuracy", ""),
                "passed_60pct": to_bool_text(best_filter.get("filtered_passed_60pct")),
                "selected_candidate": "false",
            }
        )
    return [add_meta(row, "table5_confidence_filtered_diagnostics") for row in rows]


def concentration_lookup(report_dir: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(report_dir / "vn100_ticker_concentration_summary.csv")
    lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        scope = row.get("scope", "")
        current = lookup.get(scope)
        if current is None or (to_float(row.get("contribution_share")) or 0.0) > (to_float(current.get("contribution_share")) or 0.0):
            lookup[scope] = row
    return lookup


def build_table6(artifact_dir: Path, report_dir: Path) -> list[dict[str, Any]]:
    lookup = concentration_lookup(report_dir)
    rows: list[dict[str, Any]] = []
    for frequency in ("daily", "hourly"):
        regime_rows = read_csv(artifact_dir / frequency / "regime_accuracy_summary.csv")
        best_by_regime: dict[str, dict[str, str]] = {}
        for row in regime_rows:
            regime = row.get("regime", "")
            current = best_by_regime.get(regime)
            if current is None or (to_float(row.get("accuracy")) or -1.0) > (to_float(current.get("accuracy")) or -1.0):
                best_by_regime[regime] = row
        for regime, row in sorted(best_by_regime.items()):
            scope = ""
            if frequency == "daily" and row.get("model") == "lightgbm" and row.get("horizon") == "20" and regime == "bear":
                scope = "best_regime_daily_lightgbm_h20_bear"
            if frequency == "hourly" and row.get("model") == "stacking" and row.get("horizon") == "1" and regime == "high_volatility":
                scope = "best_regime_hourly_stacking_h1_high_volatility"
            concentration = lookup.get(scope, {})
            rows.append(
                {
                    "frequency": frequency,
                    "regime": regime,
                    "best_model": row.get("model", ""),
                    "horizon": row.get("horizon", ""),
                    "n_obs": row.get("n_obs", ""),
                    "accuracy": row.get("accuracy", ""),
                    "passed_60pct": to_bool_text(row.get("passed_60pct")),
                    "reliable": to_bool_text(row.get("reliable")),
                    "top_ticker_by_contribution": concentration.get("ticker", ""),
                    "top_ticker_contribution_share": concentration.get("contribution_share", ""),
                }
            )
    return [add_meta(row, "table6_regime_specific_diagnostics") for row in rows]


def build_table7(artifact_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frequency in ("daily", "hourly"):
        for row in read_csv(artifact_dir / frequency / "significance_summary.csv"):
            rows.append(
                {
                    "frequency": frequency,
                    "model": row.get("model", ""),
                    "horizon": row.get("horizon", ""),
                    "n_obs": row.get("n_obs", ""),
                    "accuracy": row.get("accuracy", ""),
                    "null_accuracy": row.get("null_accuracy", ""),
                    "binomial_p_value": row.get("binomial_p_value", ""),
                    "bootstrap_ci_low": row.get("bootstrap_ci_low", ""),
                    "bootstrap_ci_high": row.get("bootstrap_ci_high", ""),
                    "significant_at_5pct": to_bool_text(row.get("significant_at_5pct")),
                    "significant_at_10pct": to_bool_text(row.get("significant_at_10pct")),
                }
            )
    rows.sort(key=lambda row: (row["frequency"], row["model"], to_int(row["horizon"])))
    return [add_meta(row, "table7_statistical_significance_summary") for row in rows]


def build_table8(artifact_dir: Path, report_dir: Path) -> list[dict[str, Any]]:
    daily_summary = read_json(artifact_dir / "daily" / "benchmark_summary.json")
    hourly_summary = read_json(artifact_dir / "hourly" / "benchmark_summary.json")
    sweep_rows = read_csv(artifact_dir / "confidence_threshold_sweep_summary.csv")
    daily_sweep_rows = read_csv(artifact_dir / "daily" / "confidence_threshold_sweep_summary.csv")
    rows = [
        {
            "risk_or_limitation": "Global benchmark threshold",
            "current_evidence": f"daily passed={to_bool_text(daily_summary.get('passed'))}; hourly passed={to_bool_text(hourly_summary.get('passed'))}",
            "paper_handling": "State that the global 60% benchmark did not pass.",
            "remaining_work": "Repeat after broader coverage and additional windows.",
        },
        {
            "risk_or_limitation": "Usable ticker coverage",
            "current_evidence": "Official summaries evaluate ANV, BCM, BID, BMP, BVH, BWE, CII.",
            "paper_handling": "Frame as limited-cache VN100 evidence, not a full-market conclusion.",
            "remaining_work": "Expand benchmark-usable VN100 cache coverage.",
        },
        {
            "risk_or_limitation": "Confidence sweep coverage",
            "current_evidence": f"{len(sweep_rows)} combined sweep rows; {len(daily_sweep_rows)} daily sweep data rows.",
            "paper_handling": "Mark confidence table/figure as partial.",
            "remaining_work": "Generate full daily and all-model threshold sweeps.",
        },
        {
            "risk_or_limitation": "Selected-slice concentration",
            "current_evidence": "Selected hourly confidence slice has five tickers and top-three prediction share near 79.49%.",
            "paper_handling": "Treat selected pass as narrow strategy-level diagnostic.",
            "remaining_work": "Re-test after ticker coverage broadens.",
        },
        {
            "risk_or_limitation": "Regime-specific post-hoc risk",
            "current_evidence": "Daily bear-regime h=20 rows exceed 63%, but are regime-specific diagnostics.",
            "paper_handling": "Do not describe as a stable full-market 63% method.",
            "remaining_work": "Define ex-ante regime rules and validate across windows.",
        },
        {
            "risk_or_limitation": "Trading readiness",
            "current_evidence": "Official selected slices have no cost-adjusted return, slippage, turnover, drawdown, or profit-factor artifacts.",
            "paper_handling": "State practical trading readiness is not established.",
            "remaining_work": "Run cost/slippage-aware backtests with trade and portfolio metrics.",
        },
    ]
    return [add_meta(row, "table8_robustness_limitation_matrix") for row in rows]


def write_table_pair(table_dir: Path, key: str, title: str, rows: list[dict[str, Any]], headers: list[str]) -> None:
    csv_path = table_dir / f"{key}.csv"
    md_path = table_dir / f"{key}.md"
    write_csv(csv_path, rows, headers + ["source_artifact", "claim_supported", "limitation", "status"])
    write_table_markdown(md_path, title, rows, headers, TABLE_META[key])


def write_tables(artifact_dir: Path, report_dir: Path, table_dir: Path) -> dict[str, str]:
    table_specs = {
        "table1_dataset_evaluation_scope": (
            "Table 1: Dataset and Evaluation Scope",
            build_table1(artifact_dir),
            ["item", "value", "detail"],
        ),
        "table2_model_baseline_list": (
            "Table 2: Model and Baseline List",
            build_table2(artifact_dir),
            ["type", "name", "frequency", "horizons", "role"],
        ),
        "table3_global_benchmark_results": (
            "Table 3: Global Benchmark Results",
            build_table3(artifact_dir),
            [
                "frequency",
                "overall_accuracy",
                "n_predictions",
                "best_model_accuracy",
                "best_model",
                "best_model_horizon",
                "best_baseline_accuracy",
                "best_model_delta_vs_best_baseline",
                "passed_60pct_global",
                "evaluated_tickers",
            ],
        ),
        "table4_baseline_delta_summary": (
            "Table 4: Baseline Delta Summary",
            build_table4(artifact_dir),
            [
                "frequency",
                "model",
                "horizon",
                "baseline",
                "model_accuracy",
                "baseline_accuracy",
                "accuracy_delta",
                "model_better_than_baseline",
                "model_n_obs",
            ],
        ),
        "table5_confidence_filtered_diagnostics": (
            "Table 5: Confidence-Filtered Strategy Diagnostics",
            build_table5(artifact_dir),
            [
                "scope",
                "frequency",
                "model",
                "horizon",
                "threshold",
                "total_rows",
                "evaluated_rows",
                "coverage_ratio",
                "filtered_accuracy",
                "passed_60pct",
                "selected_candidate",
            ],
        ),
        "table6_regime_specific_diagnostics": (
            "Table 6: Regime-Specific Diagnostics",
            build_table6(artifact_dir, report_dir),
            [
                "frequency",
                "regime",
                "best_model",
                "horizon",
                "n_obs",
                "accuracy",
                "passed_60pct",
                "reliable",
                "top_ticker_by_contribution",
                "top_ticker_contribution_share",
            ],
        ),
        "table7_statistical_significance_summary": (
            "Table 7: Statistical Significance Summary",
            build_table7(artifact_dir),
            [
                "frequency",
                "model",
                "horizon",
                "n_obs",
                "accuracy",
                "null_accuracy",
                "binomial_p_value",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "significant_at_5pct",
                "significant_at_10pct",
            ],
        ),
        "table8_robustness_limitation_matrix": (
            "Table 8: Robustness and Limitation Matrix",
            build_table8(artifact_dir, report_dir),
            ["risk_or_limitation", "current_evidence", "paper_handling", "remaining_work"],
        ),
    }
    statuses: dict[str, str] = {}
    for key, (title, rows, headers) in table_specs.items():
        write_table_pair(table_dir, key, title, rows, headers)
        statuses[key] = TABLE_META[key]["status"]
    return statuses


def try_import_matplotlib() -> Any | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        return plt
    except Exception:
        return None


def write_missing_figure_note(path: Path, title: str, source: str, missing_fields: str, reason: str) -> None:
    path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                f"- Source attempted: `{source}`.",
                f"- Missing fields: {missing_fields}.",
                f"- Reason: {reason}.",
                "- Status: missing.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_pipeline_figure(fig_dir: Path) -> None:
    content = [
        "# Figure 1: Research Pipeline",
        "",
        "```text",
        "VN100 official cache artifacts",
        "  -> cache usability and date/schema verification",
        "  -> train-label cutoff: target_timestamp <= 2024-12-31",
        "  -> walk-forward out-of-sample prediction on 2025 labels",
        "  -> model and baseline accuracy summaries",
        "  -> confidence-filter and regime diagnostics",
        "  -> paper tables, figures, and claim register",
        "```",
        "",
        "Source artifact: `run_config.json`, `manifest.json`, and official daily/hourly summary CSV files.",
        "",
        "Claim supported: the paper pipeline separates cache validation, train-cutoff enforcement, held-out evaluation, and diagnostic reporting.",
        "",
        "Limitation: this figure is a methodology schematic, not a new empirical result.",
        "",
        "Status: ready.",
        "",
    ]
    (fig_dir / "figure1_research_pipeline.md").write_text("\n".join(content), encoding="utf-8")


def write_walk_forward_figure(fig_dir: Path, artifact_dir: Path) -> None:
    run_config = read_json(artifact_dir / "run_config.json")
    manifest = read_json(artifact_dir / "manifest.json")
    content = [
        "# Figure 2: Walk-Forward Validation Design",
        "",
        "```text",
        f"Raw daily request:  {run_config.get('daily_start')} -> {run_config.get('daily_end')}",
        f"Raw hourly request: {run_config.get('hourly_start')} -> {run_config.get('hourly_end')}",
        "",
        f"Training labels allowed through: {run_config.get('train_cutoff')}",
        f"Cutoff rule: {run_config.get('training_label_cutoff_rule')}",
        "",
        f"Official evaluation window: {run_config.get('eval_start')} -> {run_config.get('eval_end')}",
        f"Effective daily evaluation: {manifest.get('effective_evaluation_range', {}).get('daily', {}).get('start')} -> {manifest.get('effective_evaluation_range', {}).get('daily', {}).get('end')}",
        f"Effective hourly evaluation: {manifest.get('effective_evaluation_range', {}).get('hourly', {}).get('start')} -> {manifest.get('effective_evaluation_range', {}).get('hourly', {}).get('end')}",
        "```",
        "",
        "Source artifact: `run_config.json` and `manifest.json`.",
        "",
        "Claim supported: 2025 outcomes are evaluated out of sample after a 2024-12-31 training-label cutoff.",
        "",
        "Limitation: the current evidence covers one official 2025 window.",
        "",
        "Status: ready.",
        "",
    ]
    (fig_dir / "figure2_walk_forward_design.md").write_text("\n".join(content), encoding="utf-8")


def plot_accuracy_by_model_horizon(plt: Any, artifact_dir: Path, fig_dir: Path) -> str:
    rows = []
    for frequency in ("daily", "hourly"):
        rows.extend(read_csv(artifact_dir / frequency / "classification_accuracy_summary.csv"))
    required = {"frequency", "model", "horizon", "accuracy"}
    if not rows or not required.issubset(rows[0]):
        write_missing_figure_note(
            fig_dir / "figure3_accuracy_by_model_horizon.md",
            "Figure 3: Accuracy by Model/Horizon",
            "daily/hourly classification_accuracy_summary.csv",
            ", ".join(sorted(required)),
            "Required fields are unavailable.",
        )
        return "missing"

    rows.sort(key=lambda row: (row["frequency"], to_int(row["horizon"]), row["model"]))
    labels = [f"{row['frequency'][0].upper()} h{row['horizon']} {row['model']}" for row in rows]
    values = [to_float(row["accuracy"]) or 0.0 for row in rows]
    colors = ["#4c78a8" if row["frequency"] == "daily" else "#f58518" for row in rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(rows)), values, color=colors)
    ax.axhline(0.60, color="#b22222", linewidth=1.0, linestyle="--", label="60% threshold")
    ax.axhline(0.50, color="#555555", linewidth=0.8, linestyle=":", label="50% null")
    ax.set_ylabel("Directional accuracy")
    ax.set_title("Accuracy by model, horizon, and frequency")
    ax.set_ylim(0.45, 0.62)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure3_accuracy_by_model_horizon.png", dpi=160, metadata={"Software": "matplotlib"})
    plt.close(fig)
    return "ready"


def plot_confidence_threshold(plt: Any, artifact_dir: Path, fig_dir: Path) -> str:
    rows = read_csv(artifact_dir / "confidence_threshold_sweep_summary.csv")
    required = {"threshold", "coverage_ratio", "filtered_accuracy"}
    if not rows or not required.issubset(rows[0]):
        write_missing_figure_note(
            fig_dir / "figure4_confidence_threshold_coverage_accuracy.md",
            "Figure 4: Confidence Threshold vs Coverage/Accuracy",
            "confidence_threshold_sweep_summary.csv",
            ", ".join(sorted(required)),
            "Required fields are unavailable.",
        )
        return "missing"
    rows.sort(key=lambda row: to_float(row["threshold"]) or 0.0)
    x = [to_float(row["threshold"]) or 0.0 for row in rows]
    coverage = [to_float(row["coverage_ratio"]) or 0.0 for row in rows]
    accuracy = [to_float(row["filtered_accuracy"]) or 0.0 for row in rows]
    selected = [row for row in rows if to_bool_text(row.get("selected_candidate")) == "true"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, accuracy, marker="o", linewidth=1.4, label="Filtered accuracy", color="#4c78a8")
    ax.plot(x, coverage, marker="s", linewidth=1.4, label="Coverage ratio", color="#54a24b")
    ax.axhline(0.60, color="#b22222", linewidth=1.0, linestyle="--", label="60% accuracy")
    ax.axhline(0.30, color="#666666", linewidth=0.8, linestyle=":", label="30% coverage")
    if selected:
        selected_x = to_float(selected[0]["threshold"]) or 0.0
        selected_y = to_float(selected[0]["filtered_accuracy"]) or 0.0
        ax.scatter([selected_x], [selected_y], color="#b22222", zorder=5)
        ax.annotate("selected", (selected_x, selected_y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Ratio")
    ax.set_title("Confidence threshold, coverage, and filtered accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure4_confidence_threshold_coverage_accuracy.png", dpi=160, metadata={"Software": "matplotlib"})
    plt.close(fig)
    return "partial"


def plot_regime_specific_accuracy(plt: Any, artifact_dir: Path, fig_dir: Path) -> str:
    rows = []
    for frequency in ("daily", "hourly"):
        regime_rows = read_csv(artifact_dir / frequency / "regime_accuracy_summary.csv")
        best_by_regime: dict[str, dict[str, str]] = {}
        for row in regime_rows:
            regime = row.get("regime", "")
            current = best_by_regime.get(regime)
            if current is None or (to_float(row.get("accuracy")) or -1.0) > (to_float(current.get("accuracy")) or -1.0):
                best_by_regime[regime] = row
        rows.extend(best_by_regime.values())
    required = {"frequency", "regime", "model", "horizon", "accuracy"}
    if not rows or not required.issubset(rows[0]):
        write_missing_figure_note(
            fig_dir / "figure5_regime_specific_accuracy.md",
            "Figure 5: Regime-Specific Accuracy",
            "daily/hourly regime_accuracy_summary.csv",
            ", ".join(sorted(required)),
            "Required fields are unavailable.",
        )
        return "missing"
    rows.sort(key=lambda row: (row["frequency"], row["regime"]))
    labels = [f"{row['frequency'][0].upper()} {row['regime']}\n{row['model']} h{row['horizon']}" for row in rows]
    values = [to_float(row["accuracy"]) or 0.0 for row in rows]
    colors = ["#4c78a8" if row["frequency"] == "daily" else "#f58518" for row in rows]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(range(len(rows)), values, color=colors)
    ax.axhline(0.60, color="#b22222", linewidth=1.0, linestyle="--", label="60% threshold")
    ax.set_ylabel("Best diagnostic accuracy")
    ax.set_title("Best regime-specific diagnostic accuracy by frequency/regime")
    ax.set_ylim(0.45, 0.75)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure5_regime_specific_accuracy.png", dpi=160, metadata={"Software": "matplotlib"})
    plt.close(fig)
    return "ready"


def write_figures(artifact_dir: Path, fig_dir: Path) -> dict[str, str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    write_pipeline_figure(fig_dir)
    write_walk_forward_figure(fig_dir, artifact_dir)
    statuses = {
        "figure1_research_pipeline": "ready",
        "figure2_walk_forward_design": "ready",
    }
    plt = try_import_matplotlib()
    if plt is None:
        for key, title, source, fields in [
            (
                "figure3_accuracy_by_model_horizon",
                "Figure 3: Accuracy by Model/Horizon",
                "daily/hourly classification_accuracy_summary.csv",
                "frequency, model, horizon, accuracy",
            ),
            (
                "figure4_confidence_threshold_coverage_accuracy",
                "Figure 4: Confidence Threshold vs Coverage/Accuracy",
                "confidence_threshold_sweep_summary.csv",
                "threshold, coverage_ratio, filtered_accuracy",
            ),
            (
                "figure5_regime_specific_accuracy",
                "Figure 5: Regime-Specific Accuracy",
                "daily/hourly regime_accuracy_summary.csv",
                "frequency, regime, model, horizon, accuracy",
            ),
        ]:
            write_missing_figure_note(fig_dir / f"{key}.md", title, source, fields, "matplotlib is unavailable.")
            statuses[key] = "missing"
        return statuses

    statuses["figure3_accuracy_by_model_horizon"] = plot_accuracy_by_model_horizon(plt, artifact_dir, fig_dir)
    statuses["figure4_confidence_threshold_coverage_accuracy"] = plot_confidence_threshold(plt, artifact_dir, fig_dir)
    statuses["figure5_regime_specific_accuracy"] = plot_regime_specific_accuracy(plt, artifact_dir, fig_dir)
    return statuses


def write_notes(note_dir: Path, table_statuses: dict[str, str], figure_statuses: dict[str, str]) -> None:
    note_dir.mkdir(parents=True, exist_ok=True)
    table_rows = [{"artifact": key, "status": value} for key, value in sorted(table_statuses.items())]
    figure_rows = [{"artifact": key, "status": value} for key, value in sorted(figure_statuses.items())]
    (note_dir / "paper_artifact_status.md").write_text(
        "\n".join(
            [
                "# VN100 Paper Artifact Pack Status",
                "",
                "## Tables",
                "",
                markdown_table(["artifact", "status"], table_rows),
                "",
                "## Figures",
                "",
                markdown_table(["artifact", "status"], figure_rows),
                "",
                "Status meanings: ready = artifact fields support the table or figure; partial = usable but incomplete artifact coverage; missing = required fields or rendering support absent.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (note_dir / "claim_boundary_notes.md").write_text(
        "\n".join(
            [
                "# VN100 Claim Boundary Notes",
                "",
                "- Global benchmark pass: no.",
                "- Strategy-level diagnostic pass: yes, hourly stacking h=1 at threshold 0.57 with 60.03% filtered accuracy and 31.30% coverage.",
                "- Stable full-market 63% method: no.",
                "- Regime-specific 63%+ diagnostic: yes, but bear-regime only and not a global pass.",
                "- Practical trading readiness: not established.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir
    report_dir = args.report_dir
    table_dir = report_dir / TABLE_DIR_NAME
    fig_dir = report_dir / FIGURE_DIR_NAME
    note_dir = report_dir / NOTE_DIR_NAME

    table_statuses = write_tables(artifact_dir, report_dir, table_dir)
    figure_statuses = write_figures(artifact_dir, fig_dir)
    write_notes(note_dir, table_statuses, figure_statuses)

    print(f"wrote tables to {rel(table_dir)}")
    print(f"wrote figures to {rel(fig_dir)}")
    print(f"wrote notes to {rel(note_dir)}")


if __name__ == "__main__":
    main()
