"""Create paper-ready stability outputs for the selected VN30 hourly candidate.

This audit uses existing selected-candidate artifacts only. It does not train, refit,
reselect, or regenerate row-level predictions.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.vn30_hourly_dual_track_common import LOCKED_RF_H60, REPO_ROOT, markdown_table, pct, write_csv  # noqa: E402

SOURCE_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_track_a_target62_validation_safe"
ROBUSTNESS_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_target62_stability_robustness"
OUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_target62_paper_ready_stability"
BASELINE_LOGISTIC_H40 = 0.6043200785468826
TARGET62 = 0.62


def read_one(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    return frame.iloc[0].to_dict() if not frame.empty else {}


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def normal_two_sided_p(z: float) -> float:
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def binomial_normal_test(successes: int, n: int, p0: float, label: str) -> dict[str, Any]:
    if n <= 0 or p0 <= 0 or p0 >= 1:
        return {"test": label, "successes": successes, "n": n, "null_p": p0, "z_score": math.nan, "p_value_two_sided": math.nan, "result": "not_practical"}
    z = (successes - n * p0) / math.sqrt(n * p0 * (1.0 - p0))
    p_value = normal_two_sided_p(z)
    return {"test": label, "successes": successes, "n": n, "null_p": p0, "z_score": z, "p_value_two_sided": p_value, "result": "significant" if p_value < 0.05 else "not_significant"}


def bootstrap_from_tickers(by_ticker: pd.DataFrame, iterations: int = 20000) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    rows = pd.to_numeric(by_ticker["rows"], errors="coerce").to_numpy(dtype=float)
    acc = pd.to_numeric(by_ticker["accuracy"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(rows) & np.isfinite(acc) & (rows > 0)
    rows = rows[valid]
    acc = acc[valid]
    sampled = rng.integers(0, len(rows), size=(iterations, len(rows)))
    sample_rows = rows[sampled]
    sample_acc = acc[sampled]
    weighted = (sample_rows * sample_acc).sum(axis=1) / sample_rows.sum(axis=1)
    return {
        "bootstrap_source": "ticker_weighted_resample",
        "iterations": iterations,
        "ci_low": float(np.quantile(weighted, 0.025)),
        "ci_high": float(np.quantile(weighted, 0.975)),
        "bootstrap_mean": float(np.mean(weighted)),
    }


def slice_stats(frame: pd.DataFrame, name: str, selected_accuracy: float) -> dict[str, Any]:
    acc = pd.to_numeric(frame["accuracy"], errors="coerce")
    return {
        f"mean_{name}_accuracy": float(acc.mean()),
        f"median_{name}_accuracy": float(acc.median()),
        f"min_{name}_accuracy": float(acc.min()),
        f"max_{name}_accuracy": float(acc.max()),
        f"{name}_accuracy_std": float(acc.std(ddof=0)),
        f"{name}s_above_50": int((acc >= 0.50).sum()),
        f"{name}s_above_55": int((acc >= 0.55).sum()),
        f"{name}s_above_60": int((acc >= 0.60).sum()),
        f"{name}s_above_61_51": int((acc >= selected_accuracy).sum()),
        f"{name}s_above_62": int((acc >= TARGET62).sum()),
        f"{name}s_above_65": int((acc >= 0.65).sum()),
        f"{name}_count": int(len(frame)),
    }


def classify_component(frame: pd.DataFrame, name: str) -> str:
    acc = pd.to_numeric(frame["accuracy"], errors="coerce")
    share60 = float((acc >= 0.60).mean()) if len(acc) else 0.0
    share62 = float((acc >= TARGET62).mean()) if len(acc) else 0.0
    if share62 >= 0.70:
        return f"{name}_stable"
    if share60 >= 0.60:
        return f"{name}_moderately_stable"
    if share60 >= 0.40:
        return f"{name}_concentrated_or_mixed"
    return f"{name}_unstable"


def rolling_unavailable(window: int) -> list[dict[str, Any]]:
    return [
        {
            "window_rows": window,
            "status": "unavailable",
            "reason": "row_level_predictions_not_saved; audit does not regenerate predictions or train models",
            "rolling_accuracy": "",
            "rolling_majority_baseline": "",
            "rolling_lift": "",
        }
    ]


def expanding_period(frame: pd.DataFrame, period_col: str) -> list[dict[str, Any]]:
    ordered = frame.sort_values(period_col).copy()
    out: list[dict[str, Any]] = []
    total_rows = 0.0
    total_correct = 0.0
    total_baseline_correct = 0.0
    for _, row in ordered.iterrows():
        rows = as_float(row["rows"])
        acc = as_float(row["accuracy"])
        baseline = as_float(row["majority_baseline_accuracy"])
        total_rows += rows
        total_correct += rows * acc
        total_baseline_correct += rows * baseline
        expanding_acc = total_correct / total_rows if total_rows else math.nan
        expanding_base = total_baseline_correct / total_rows if total_rows else math.nan
        out.append(
            {
                "period": row[period_col],
                "rows": int(rows),
                "period_accuracy": acc,
                "period_majority_baseline": baseline,
                "period_lift": acc - baseline,
                "expanding_accuracy": expanding_acc,
                "expanding_majority_baseline": expanding_base,
                "expanding_lift": expanding_acc - expanding_base,
            }
        )
    return out


def paper_classification(final_accuracy: float, ticker_class: str, time_class: str, regime_class: str, mismatch: str) -> tuple[str, str]:
    if final_accuracy >= 0.65 and "stable" in ticker_class and "stable" in time_class and mismatch != "high_positive_final_gap":
        return "stable", "final65"
    if final_accuracy >= TARGET62 and "unstable" not in time_class and mismatch != "high_positive_final_gap":
        return "moderately_stable", "target62"
    if final_accuracy >= BASELINE_LOGISTIC_H40 and "unstable" not in ticker_class:
        return "concentrated_or_mixed", "improved_baseline60"
    return "unstable", "no_success_claim"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = read_one(SOURCE_DIR / "selected_candidate_summary.csv")
    mismatch = read_one(SOURCE_DIR / "target62_validation_mismatch.csv")
    by_ticker = pd.read_csv(SOURCE_DIR / "target62_by_ticker.csv")
    by_time = pd.read_csv(SOURCE_DIR / "target62_by_time.csv")
    by_regime = pd.read_csv(SOURCE_DIR / "target62_by_regime.csv")
    by_month = by_time[by_time["time_grain"].astype(str) == "month"].copy()
    by_quarter = by_time[by_time["time_grain"].astype(str) == "quarter"].copy()

    final_accuracy = as_float(selected["final_accuracy"])
    final_rows = int(as_float(selected["final_rows"]))
    correct = int(round(final_accuracy * final_rows))
    incorrect = final_rows - correct
    majority_baseline = as_float(selected["final_baseline_accuracy"])
    baseline_correct = int(round(majority_baseline * final_rows))
    bootstrap = bootstrap_from_tickers(by_ticker)
    standard_error = math.sqrt(final_accuracy * (1.0 - final_accuracy) / final_rows)
    sig_rows = [
        binomial_normal_test(correct, final_rows, 0.50, "vs_50_percent"),
        binomial_normal_test(correct, final_rows, majority_baseline, "vs_majority_baseline"),
    ]
    bootstrap_row = {
        **bootstrap,
        "standard_error": standard_error,
        "target62_inside_ci": bool(bootstrap["ci_low"] <= TARGET62 <= bootstrap["ci_high"]),
        "baseline60_43_inside_ci": bool(bootstrap["ci_low"] <= BASELINE_LOGISTIC_H40 <= bootstrap["ci_high"]),
    }
    ticker_stats = slice_stats(by_ticker, "ticker", final_accuracy)
    month_stats = slice_stats(by_month, "month", final_accuracy)
    quarter_stats = slice_stats(by_quarter, "quarter", final_accuracy)
    regime_acc = pd.to_numeric(by_regime["accuracy"], errors="coerce")
    regime_stats = {
        "regime_count": int(len(by_regime)),
        "regimes_above_60": int((regime_acc >= 0.60).sum()),
        "regimes_above_62": int((regime_acc >= TARGET62).sum()),
        "regimes_with_positive_lift": int((pd.to_numeric(by_regime["delta_vs_majority_baseline"], errors="coerce") > 0).sum()),
    }
    ticker_class = classify_component(by_ticker, "ticker")
    month_class = classify_component(by_month, "time")
    quarter_class = classify_component(by_quarter, "quarter")
    regime_class = classify_component(by_regime, "regime")
    time_class = "time_concentrated_or_mixed" if "unstable" not in {month_class, quarter_class} and ("mixed" in month_class or "mixed" in quarter_class) else ("time_unstable" if "unstable" in {month_class, quarter_class} else "time_moderately_stable")
    stability, claim = paper_classification(final_accuracy, ticker_class, time_class, regime_class, str(mismatch.get("validation_final_mismatch", "")))

    global_summary = {
        "model": selected.get("model", ""),
        "horizon": selected.get("horizon", ""),
        "feature_set": selected.get("feature_set", ""),
        "threshold": selected.get("threshold", ""),
        "final_accuracy": final_accuracy,
        "total_rows": final_rows,
        "correct_predictions": correct,
        "incorrect_predictions": incorrect,
        "majority_baseline_accuracy": majority_baseline,
        "always_up_baseline_accuracy": "",
        "previous_direction_baseline_accuracy": "",
        "delta_vs_majority_baseline": as_float(selected["final_delta_vs_baseline"]),
        "delta_vs_60_43": as_float(selected["delta_vs_60_43"]),
        "delta_vs_60_31": final_accuracy - LOCKED_RF_H60,
        "pass_60": bool(final_accuracy >= 0.60),
        "pass_60_43": bool(final_accuracy > BASELINE_LOGISTIC_H40),
        "pass_62": bool(final_accuracy >= TARGET62),
        "pass_65": bool(final_accuracy >= 0.65),
        "validation_accuracy": as_float(mismatch.get("validation_accuracy")),
        "validation_final_gap": as_float(mismatch.get("validation_final_gap")),
        "validation_final_mismatch": mismatch.get("validation_final_mismatch", ""),
        "stability_classification": stability,
        "paper_ready_claim_level": claim,
        **ticker_stats,
        **month_stats,
        **quarter_stats,
        **regime_stats,
    }

    paper_summary = {
        "selected_candidate": f"{selected.get('model')} h={selected.get('horizon')} {selected.get('feature_set')} threshold={selected.get('threshold')}",
        "final_accuracy": final_accuracy,
        "mean_ticker_accuracy": ticker_stats["mean_ticker_accuracy"],
        "median_ticker_accuracy": ticker_stats["median_ticker_accuracy"],
        "mean_month_accuracy": month_stats["mean_month_accuracy"],
        "median_month_accuracy": month_stats["median_month_accuracy"],
        "mean_quarter_accuracy": quarter_stats["mean_quarter_accuracy"],
        "median_quarter_accuracy": quarter_stats["median_quarter_accuracy"],
        "majority_baseline_lift": as_float(selected["final_delta_vs_baseline"]),
        "bootstrap_ci_low": bootstrap["ci_low"],
        "bootstrap_ci_high": bootstrap["ci_high"],
        "significance_result": "significant_vs_50_and_majority_baseline" if all(row["result"] == "significant" for row in sig_rows) else "not_significant_on_all_tests",
        "validation_final_gap": as_float(mismatch.get("validation_final_gap")),
        "ticker_stability_classification": ticker_class,
        "time_stability_classification": time_class,
        "regime_stability_classification": regime_class,
        "stability_classification": stability,
        "claim_level": claim,
        "target62_claim": False,
        "final65_claim": False,
    }

    figure_index = [
        {"figure_id": "fig_ticker_accuracy", "source_csv": "by_ticker.csv", "description": "Ticker-level final accuracy and majority-baseline lift"},
        {"figure_id": "fig_month_accuracy", "source_csv": "by_month.csv", "description": "Monthly final accuracy and majority-baseline lift"},
        {"figure_id": "fig_quarter_accuracy", "source_csv": "by_quarter.csv", "description": "Quarterly final accuracy and majority-baseline lift"},
        {"figure_id": "fig_regime_accuracy", "source_csv": "by_regime.csv", "description": "Regime-level final accuracy and majority-baseline lift"},
        {"figure_id": "fig_monthly_expanding", "source_csv": "rolling_monthly_expanding.csv", "description": "Monthly expanding accuracy"},
        {"figure_id": "fig_quarterly_expanding", "source_csv": "rolling_quarterly_expanding.csv", "description": "Quarterly expanding accuracy"},
        {"figure_id": "fig_rolling_rows", "source_csv": "rolling_accuracy_250/500/1000.csv", "description": "Unavailable: row-level predictions not saved"},
    ]

    write_csv(OUT_DIR / "global_summary.csv", [global_summary])
    by_ticker.to_csv(OUT_DIR / "by_ticker.csv", index=False)
    by_month.to_csv(OUT_DIR / "by_month.csv", index=False)
    by_quarter.to_csv(OUT_DIR / "by_quarter.csv", index=False)
    by_regime.to_csv(OUT_DIR / "by_regime.csv", index=False)
    write_csv(OUT_DIR / "rolling_accuracy_250.csv", rolling_unavailable(250))
    write_csv(OUT_DIR / "rolling_accuracy_500.csv", rolling_unavailable(500))
    write_csv(OUT_DIR / "rolling_accuracy_1000.csv", rolling_unavailable(1000))
    write_csv(OUT_DIR / "rolling_monthly_expanding.csv", expanding_period(by_month, "month"))
    write_csv(OUT_DIR / "rolling_quarterly_expanding.csv", expanding_period(by_quarter, "quarter"))
    write_csv(OUT_DIR / "bootstrap_ci.csv", [bootstrap_row])
    write_csv(OUT_DIR / "significance_tests.csv", sig_rows)
    write_csv(OUT_DIR / "validation_final_mismatch.csv", [mismatch])
    write_csv(OUT_DIR / "paper_ready_summary_table.csv", [paper_summary])
    write_csv(OUT_DIR / "paper_ready_figure_index.csv", figure_index)

    report = [
        "# VN30 Hourly Target62 Paper-Ready Stability Audit",
        "",
        f"- Selected candidate: `{paper_summary['selected_candidate']}`.",
        f"- Final accuracy: {pct(final_accuracy)}.",
        f"- Majority baseline lift: {pct(selected.get('final_delta_vs_baseline'))}.",
        f"- Mean ticker accuracy: {pct(ticker_stats['mean_ticker_accuracy'])}; median: {pct(ticker_stats['median_ticker_accuracy'])}.",
        f"- Mean month accuracy: {pct(month_stats['mean_month_accuracy'])}; median: {pct(month_stats['median_month_accuracy'])}.",
        f"- Mean quarter accuracy: {pct(quarter_stats['mean_quarter_accuracy'])}; median: {pct(quarter_stats['median_quarter_accuracy'])}.",
        f"- Bootstrap CI: {pct(bootstrap['ci_low'])} to {pct(bootstrap['ci_high'])}.",
        f"- Significance: {paper_summary['significance_result']}.",
        f"- Validation-final mismatch: `{mismatch.get('validation_final_mismatch')}` gap={pct(mismatch.get('validation_final_gap'))}.",
        f"- Stability classification: `{stability}`.",
        f"- Claim level: `{claim}`.",
        "",
        "## Rolling Accuracy Interpretation",
        "",
        "Row-level rolling windows of 250, 500, and 1000 rows are unavailable because row-level predictions were not saved in the target62 run and this audit does not regenerate predictions. Monthly and quarterly expanding accuracy tables are provided for paper figures.",
        "",
        "## Regime Stability Interpretation",
        "",
        markdown_table(["market_regime_v2", "rows", "accuracy", "majority_baseline_accuracy", "delta_vs_majority_baseline"], by_regime.to_dict("records")),
        "",
        "## Claim Boundary",
        "",
        "The result can support an exploratory improved baseline60 statement. It cannot support target62 because final accuracy is below 62%, and it cannot support final65. The paper must disclose mixed time and regime stability and the high positive validation-final gap.",
        "",
        "## Recommended Paper Wording",
        "",
        "Under the Track A canonical-like VN30 hourly setup, a pre-registered validation-selected L2 Logistic h40 model achieved 61.51% final pooled directional accuracy with full 30-stock coverage, exceeding the 60.43% Logistic h40 baseline by 1.08 percentage points. However, the result did not reach the 62% target and showed mixed time stability with a high positive validation-final gap, so it is reported as exploratory improved baseline60 evidence rather than target62 or final65 evidence.",
        "",
    ]
    (OUT_DIR / "paper_ready_stability_audit.md").write_text("\n".join(report), encoding="utf-8")
    print(f"paper_ready_stability_status=completed stability={stability} claim={claim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
