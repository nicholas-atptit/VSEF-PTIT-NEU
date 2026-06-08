"""Audit stability and robustness of the selected Track A candidate.

This script uses existing selected-candidate audit artifacts only. It does not train,
refit, or reselect models.
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

INPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_track_a_target62_validation_safe"
OUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_target62_stability_robustness"
BASELINE_LOGISTIC_H40 = 0.6043200785468826
TARGET62 = 0.62


def read_one(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    return frame.iloc[0].to_dict() if not frame.empty else {}


def normal_two_sided_p(z: float) -> float:
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def binomial_normal_test(successes: int, n: int, p0: float) -> dict[str, Any]:
    if n <= 0 or p0 <= 0 or p0 >= 1:
        return {"successes": successes, "n": n, "null_p": p0, "z_score": math.nan, "p_value_two_sided": math.nan}
    mean = n * p0
    variance = n * p0 * (1.0 - p0)
    z = (successes - mean) / math.sqrt(variance)
    return {"successes": successes, "n": n, "null_p": p0, "z_score": z, "p_value_two_sided": normal_two_sided_p(z)}


def bootstrap_ci_from_slices(frame: pd.DataFrame, iterations: int = 20000) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    rows = pd.to_numeric(frame["rows"], errors="coerce").to_numpy(dtype=float)
    acc = pd.to_numeric(frame["accuracy"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(rows) & np.isfinite(acc) & (rows > 0)
    rows = rows[valid]
    acc = acc[valid]
    if len(rows) == 0:
        return {"bootstrap_source": "ticker_weighted", "iterations": iterations, "ci_low": math.nan, "ci_high": math.nan, "bootstrap_mean": math.nan}
    sampled = rng.integers(0, len(rows), size=(iterations, len(rows)))
    sample_rows = rows[sampled]
    sample_acc = acc[sampled]
    weighted = (sample_rows * sample_acc).sum(axis=1) / sample_rows.sum(axis=1)
    return {
        "bootstrap_source": "ticker_weighted",
        "iterations": iterations,
        "ci_low": float(np.quantile(weighted, 0.025)),
        "ci_high": float(np.quantile(weighted, 0.975)),
        "bootstrap_mean": float(np.mean(weighted)),
    }


def describe_slice(frame: pd.DataFrame, prefix: str, threshold: float) -> dict[str, Any]:
    acc = pd.to_numeric(frame["accuracy"], errors="coerce")
    rows = {
        f"mean_{prefix}_accuracy": float(acc.mean()),
        f"median_{prefix}_accuracy": float(acc.median()),
        f"{prefix}_accuracy_std": float(acc.std(ddof=0)),
        f"{prefix}s_above_60": int((acc >= 0.60).sum()),
        f"{prefix}s_above_selected_accuracy": int((acc >= threshold).sum()),
        f"{prefix}s_above_62": int((acc >= TARGET62).sum()),
        f"{prefix}_count": int(len(frame)),
    }
    return rows


def rolling_accuracy(months: pd.DataFrame) -> list[dict[str, Any]]:
    month_rows = months.copy()
    month_rows["period"] = month_rows["month"].astype(str)
    month_rows = month_rows.sort_values("period")
    rows: list[dict[str, Any]] = []
    for idx, row in month_rows.iterrows():
        upto = month_rows.loc[:idx]
        last3 = month_rows.iloc[max(0, len(rows) - 2) : len(rows) + 1]
        rows.append(
            {
                "period": row["period"],
                "rows": int(row["rows"]),
                "accuracy": float(row["accuracy"]),
                "majority_baseline_accuracy": float(row["majority_baseline_accuracy"]),
                "delta_vs_majority_baseline": float(row["delta_vs_majority_baseline"]),
                "expanding_weighted_accuracy": float((pd.to_numeric(upto["rows"]) * pd.to_numeric(upto["accuracy"])).sum() / pd.to_numeric(upto["rows"]).sum()),
                "rolling_3_period_weighted_accuracy": float((pd.to_numeric(last3["rows"]) * pd.to_numeric(last3["accuracy"])).sum() / pd.to_numeric(last3["rows"]).sum()),
            }
        )
    return rows


def max_wrong_sequence_proxy(months: pd.DataFrame) -> dict[str, Any]:
    ordered = months.sort_values("month").copy()
    ordered["below_50"] = pd.to_numeric(ordered["accuracy"], errors="coerce") < 0.50
    longest = 0
    current = 0
    for value in ordered["below_50"]:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return {"drawdown_proxy": "consecutive_months_below_50_accuracy", "max_consecutive_months_below_50": longest}


def classify(global_summary: dict[str, Any], ticker_stats: dict[str, Any], month_stats: dict[str, Any], quarter_stats: dict[str, Any], regime: pd.DataFrame, mismatch: str) -> str:
    final_accuracy = float(global_summary["final_accuracy"])
    ticker_share = int(ticker_stats["tickers_above_60"]) / max(int(ticker_stats["ticker_count"]), 1)
    month_share = int(month_stats["months_above_60"]) / max(int(month_stats["month_count"]), 1)
    quarter_share = int(quarter_stats["quarters_above_60"]) / max(int(quarter_stats["quarter_count"]), 1)
    regime_positive = int((pd.to_numeric(regime["delta_vs_majority_baseline"], errors="coerce") > 0).sum())
    regime_share = regime_positive / max(len(regime), 1)
    if final_accuracy >= TARGET62 and ticker_share >= 0.70 and month_share >= 0.70 and quarter_share >= 0.60 and regime_share >= 0.75 and mismatch != "high_positive_final_gap":
        return "stable"
    if final_accuracy >= 0.60 and ticker_share >= 0.60 and month_share >= 0.45 and quarter_share >= 0.40 and regime_share >= 0.50:
        return "moderately_stable" if mismatch != "high_positive_final_gap" else "concentrated_or_mixed"
    if final_accuracy >= 0.60 and ticker_share >= 0.50:
        return "concentrated_or_mixed"
    return "unstable"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = read_one(INPUT_DIR / "selected_candidate_summary.csv")
    mismatch = read_one(INPUT_DIR / "target62_validation_mismatch.csv")
    by_ticker = pd.read_csv(INPUT_DIR / "target62_by_ticker.csv")
    by_time = pd.read_csv(INPUT_DIR / "target62_by_time.csv")
    by_regime = pd.read_csv(INPUT_DIR / "target62_by_regime.csv")
    by_month = by_time[by_time["time_grain"].astype(str) == "month"].copy()
    by_quarter = by_time[by_time["time_grain"].astype(str) == "quarter"].copy()

    final_accuracy = float(selected["final_accuracy"])
    final_rows = int(float(selected["final_rows"]))
    final_baseline = float(selected["final_baseline_accuracy"])
    successes = int(round(final_accuracy * final_rows))
    baseline_successes = int(round(final_baseline * final_rows))
    ticker_stats = describe_slice(by_ticker, "ticker", final_accuracy)
    month_stats = describe_slice(by_month, "month", final_accuracy)
    quarter_stats = describe_slice(by_quarter, "quarter", final_accuracy)
    regime_acc = pd.to_numeric(by_regime["accuracy"], errors="coerce")
    regime_stats = {
        "regime_count": int(len(by_regime)),
        "regimes_with_positive_lift": int((pd.to_numeric(by_regime["delta_vs_majority_baseline"], errors="coerce") > 0).sum()),
        "regimes_above_60": int((regime_acc >= 0.60).sum()),
        "regimes_above_62": int((regime_acc >= TARGET62).sum()),
    }
    bootstrap = bootstrap_ci_from_slices(by_ticker)
    test_50 = binomial_normal_test(successes, final_rows, 0.50)
    test_baseline = binomial_normal_test(successes, final_rows, final_baseline)
    wrong_proxy = max_wrong_sequence_proxy(by_month)
    classification = classify(selected, ticker_stats, month_stats, quarter_stats, by_regime, str(mismatch.get("validation_final_mismatch", "")))

    global_summary = {
        "model": selected.get("model", ""),
        "horizon": selected.get("horizon", ""),
        "feature_set": selected.get("feature_set", ""),
        "threshold": selected.get("threshold", ""),
        "final_accuracy": final_accuracy,
        "final_rows": final_rows,
        "final_coverage": float(selected["final_coverage"]),
        "active_ticker_count": int(float(selected["active_ticker_count"])),
        "majority_baseline_accuracy": final_baseline,
        "delta_vs_majority_baseline": float(selected["final_delta_vs_baseline"]),
        "delta_vs_60_43": float(selected["delta_vs_60_43"]),
        "delta_vs_60_31": final_accuracy - LOCKED_RF_H60,
        "successes": successes,
        "majority_baseline_successes": baseline_successes,
        "validation_final_gap": float(mismatch.get("validation_final_gap", math.nan)),
        "validation_final_mismatch": mismatch.get("validation_final_mismatch", ""),
        "stability_classification": classification,
        **ticker_stats,
        **month_stats,
        **quarter_stats,
        **regime_stats,
        **wrong_proxy,
    }

    bootstrap_rows = [
        {
            **bootstrap,
            "binomial_vs_50_z": test_50["z_score"],
            "binomial_vs_50_p_value_two_sided": test_50["p_value_two_sided"],
            "binomial_vs_majority_baseline_z": test_baseline["z_score"],
            "binomial_vs_majority_baseline_p_value_two_sided": test_baseline["p_value_two_sided"],
            "significance_summary": "significant_vs_50_and_majority_baseline" if test_50["p_value_two_sided"] < 0.05 and test_baseline["p_value_two_sided"] < 0.05 else "not_significant_on_all_tests",
        }
    ]

    write_csv(OUT_DIR / "global_summary.csv", [global_summary])
    by_ticker.to_csv(OUT_DIR / "by_ticker.csv", index=False)
    by_month.to_csv(OUT_DIR / "by_month.csv", index=False)
    by_quarter.to_csv(OUT_DIR / "by_quarter.csv", index=False)
    by_regime.to_csv(OUT_DIR / "by_regime.csv", index=False)
    write_csv(OUT_DIR / "bootstrap_ci.csv", bootstrap_rows)
    write_csv(OUT_DIR / "rolling_accuracy.csv", rolling_accuracy(by_month))

    worst_tickers = by_ticker.sort_values("accuracy").head(5).to_dict("records")
    best_tickers = by_ticker.sort_values("accuracy", ascending=False).head(5).to_dict("records")
    worst_months = by_month.sort_values("accuracy").head(5).to_dict("records")
    worst_quarters = by_quarter.sort_values("accuracy").head(5).to_dict("records")
    report = [
        "# VN30 Hourly Target62 Stability Robustness Report",
        "",
        f"- Selected candidate: `{selected.get('model')}` h={selected.get('horizon')} `{selected.get('feature_set')}` threshold={selected.get('threshold')}.",
        f"- Final accuracy: {pct(final_accuracy)}.",
        f"- Final rows: {final_rows}.",
        f"- Final coverage: {pct(selected.get('final_coverage'))}.",
        f"- Active ticker count: {selected.get('active_ticker_count')}.",
        f"- Majority baseline: {pct(final_baseline)}.",
        f"- Delta vs majority baseline: {pct(selected.get('final_delta_vs_baseline'))}.",
        f"- Delta vs 60.43: {pct(selected.get('delta_vs_60_43'))}.",
        f"- Delta vs 60.31: {pct(final_accuracy - LOCKED_RF_H60)}.",
        f"- Mean ticker accuracy: {pct(ticker_stats['mean_ticker_accuracy'])}.",
        f"- Median ticker accuracy: {pct(ticker_stats['median_ticker_accuracy'])}.",
        f"- Mean month accuracy: {pct(month_stats['mean_month_accuracy'])}.",
        f"- Median month accuracy: {pct(month_stats['median_month_accuracy'])}.",
        f"- Mean quarter accuracy: {pct(quarter_stats['mean_quarter_accuracy'])}.",
        f"- Median quarter accuracy: {pct(quarter_stats['median_quarter_accuracy'])}.",
        f"- Bootstrap CI: {pct(bootstrap['ci_low'])} to {pct(bootstrap['ci_high'])}.",
        f"- Significance test: {bootstrap_rows[0]['significance_summary']}.",
        f"- Validation-final mismatch: `{mismatch.get('validation_final_mismatch')}` gap={pct(mismatch.get('validation_final_gap'))}.",
        f"- Stability classification: `{classification}`.",
        "",
        "## Regime Stability",
        "",
        markdown_table(["market_regime_v2", "rows", "accuracy", "majority_baseline_accuracy", "delta_vs_majority_baseline"], by_regime.to_dict("records")),
        "",
        "## Worst Tickers",
        "",
        markdown_table(["ticker", "rows", "accuracy", "majority_baseline_accuracy", "delta_vs_majority_baseline"], worst_tickers),
        "",
        "## Best Tickers",
        "",
        markdown_table(["ticker", "rows", "accuracy", "majority_baseline_accuracy", "delta_vs_majority_baseline"], best_tickers),
        "",
        "## Worst Months",
        "",
        markdown_table(["month", "rows", "accuracy", "majority_baseline_accuracy", "delta_vs_majority_baseline"], worst_months),
        "",
        "## Worst Quarters",
        "",
        markdown_table(["quarter", "rows", "accuracy", "majority_baseline_accuracy", "delta_vs_majority_baseline"], worst_quarters),
        "",
    ]
    (OUT_DIR / "stability_robustness_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"target62_stability_status=completed classification={classification} final_accuracy={final_accuracy:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
