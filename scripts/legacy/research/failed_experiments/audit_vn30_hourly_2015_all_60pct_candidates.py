"""Exhaustive >60% accuracy audit from existing VN30 hourly 2015 benchmark predictions."""

from __future__ import annotations

import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PREDICTIONS_PATH = (
    REPO_ROOT / "outputs" / "vn30_hourly_2015_jan2025_benchmark" / "hourly" / "predicted_vs_actual.csv"
)
REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_benchmark" / "above60"
CSV_PATH = REPORT_ROOT / "vn30_all_60pct_candidates.csv"
MD_PATH = REPORT_ROOT / "vn30_all_60pct_candidates.md"
SUMMARY_PATH = REPORT_ROOT / "vn30_all_60pct_candidates_summary.md"

CONFIDENCE_THRESHOLDS = [round(0.50 + i * 0.025, 3) for i in range(19)]
COVERAGE_FLOORS = [0.50, 0.40, 0.30, 0.20, 0.10]
MIN_ROW_FLOORS = [100, 300, 500, 1000]
TICKER_MIN_OBS = [50, 100, 200, 500]

CANDIDATE_COLUMNS = [
    "candidate_type",
    "model",
    "horizon",
    "filter_description",
    "observations",
    "coverage_ratio",
    "accuracy",
    "correct_count",
    "incorrect_count",
    "pass_60pct",
    "min_row_floor_met",
    "coverage_floor_met",
    "post_hoc_warning",
    "allowed_claim_level",
]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fmt_pct(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    return f"{numeric * 100:.2f}%"


def bool_text(value: bool) -> str:
    return "yes" if value else "no"


def markdown_table(headers: list[str], rows: list[dict[str, Any]], *, max_rows: int | None = None) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    display_rows = rows if max_rows is None else rows[:max_rows]
    for row in display_rows:
        values = [str(row.get(h, "")).replace("|", "\\|") for h in headers]
        lines.append("| " + " | ".join(values) + " |")
    if max_rows is not None and len(rows) > max_rows:
        lines.append("| " + " | ".join(["..."] + [""] * (len(headers) - 1)) + " |")
    return "\n".join(lines)


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions: {rel(path)}")
    df = pd.read_csv(path, low_memory=False)
    df = df[df["frequency"].astype(str).str.lower().eq("hourly")].copy()
    df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce").astype(int)
    df = df[df["is_correct"].isin([0, 1])].copy()
    return df


def make_candidate(
    candidate_type: str,
    model: str,
    horizon: int,
    filter_description: str,
    observations: int,
    coverage_ratio: float,
    accuracy: float,
    correct_count: int,
    incorrect_count: int,
    pass_60pct: bool,
    min_row_floor_met: bool,
    coverage_floor_met: bool,
    post_hoc_warning: bool,
    allowed_claim_level: str,
) -> dict[str, Any]:
    return {
        "candidate_type": candidate_type,
        "model": model,
        "horizon": horizon,
        "filter_description": filter_description,
        "observations": observations,
        "coverage_ratio": round(coverage_ratio, 4),
        "accuracy": round(accuracy, 6),
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "pass_60pct": bool_text(pass_60pct),
        "min_row_floor_met": bool_text(min_row_floor_met),
        "coverage_floor_met": bool_text(coverage_floor_met),
        "post_hoc_warning": bool_text(post_hoc_warning),
        "allowed_claim_level": allowed_claim_level,
    }


def audit_global(df: pd.DataFrame) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    total = len(df)

    for model in sorted(df["model"].unique()):
        m_df = df[df["model"] == model]
        acc = m_df["is_correct"].mean()
        correct = int(m_df["is_correct"].sum())
        incorrect = len(m_df) - correct
        candidates.append(make_candidate(
            "global_model", model, 0, f"model={model}",
            len(m_df), 1.0, acc, correct, incorrect,
            acc >= 0.60, True, True, False, "global",
        ))

        for horizon in sorted(m_df["horizon"].unique()):
            mh_df = m_df[m_df["horizon"] == horizon]
            acc = mh_df["is_correct"].mean()
            correct = int(mh_df["is_correct"].sum())
            incorrect = len(mh_df) - correct
            cov = len(mh_df) / total if total else 0.0
            candidates.append(make_candidate(
                "global_model_horizon", model, horizon, f"model={model}, horizon={horizon}",
                len(mh_df), cov, acc, correct, incorrect,
                acc >= 0.60, True, True, False, "global",
            ))

    return candidates


def audit_ticker(df: pd.DataFrame) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    total = len(df)

    for model in sorted(df["model"].unique()):
        m_df = df[df["model"] == model]
        for horizon in sorted(m_df["horizon"].unique()):
            mh_df = m_df[m_df["horizon"] == horizon]
            for ticker in sorted(mh_df["ticker"].unique()):
                t_df = mh_df[mh_df["ticker"] == ticker]
                n = len(t_df)
                if n < 50:
                    continue
                acc = t_df["is_correct"].mean()
                correct = int(t_df["is_correct"].sum())
                incorrect = n - correct
                cov = n / total if total else 0.0
                for min_obs in TICKER_MIN_OBS:
                    row_ok = n >= min_obs
                    candidates.append(make_candidate(
                        "model_horizon_ticker", model, horizon,
                        f"model={model}, horizon={horizon}, ticker={ticker}, min_obs>={min_obs}",
                        n, cov, acc, correct, incorrect,
                        acc >= 0.60, row_ok, True, True, "exploratory",
                    ))

    return candidates


def audit_confidence(df: pd.DataFrame) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    total = len(df)

    for model in sorted(df["model"].unique()):
        m_df = df[df["model"] == model]
        for horizon in sorted(m_df["horizon"].unique()):
            mh_df = m_df[m_df["horizon"] == horizon]
            parent_total = len(mh_df)
            if parent_total == 0:
                continue
            for thresh in CONFIDENCE_THRESHOLDS:
                sel = mh_df[mh_df["confidence"] >= thresh]
                n = len(sel)
                if n == 0:
                    continue
                acc = sel["is_correct"].mean()
                correct = int(sel["is_correct"].sum())
                incorrect = n - correct
                cov = n / parent_total
                for min_rows in MIN_ROW_FLOORS:
                    for cov_floor in COVERAGE_FLOORS:
                        row_ok = n >= min_rows
                        cov_ok = cov >= cov_floor
                        pass_60 = acc >= 0.60
                        claim = "conditional" if (row_ok and cov_ok) else "exploratory"
                        candidates.append(make_candidate(
                            "model_horizon_confidence", model, horizon,
                            f"model={model}, horizon={horizon}, conf>={thresh}, min_rows>={min_rows}, cov_floor>={cov_floor}",
                            n, cov, acc, correct, incorrect,
                            pass_60, row_ok, cov_ok, True, claim,
                        ))

    return candidates


def audit_regime(df: pd.DataFrame) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    total = len(df)

    regime_cols = []
    if "regime" in df.columns:
        regime_cols.append("regime")
    if "volatility_regime" in df.columns:
        regime_cols.append("volatility_regime")

    for reg_col in regime_cols:
        for model in sorted(df["model"].unique()):
            m_df = df[df["model"] == model]
            for horizon in sorted(m_df["horizon"].unique()):
                mh_df = m_df[m_df["horizon"] == horizon]
                for regime_val in sorted(mh_df[reg_col].dropna().unique()):
                    r_df = mh_df[mh_df[reg_col] == regime_val]
                    n = len(r_df)
                    if n < 50:
                        continue
                    acc = r_df["is_correct"].mean()
                    correct = int(r_df["is_correct"].sum())
                    incorrect = n - correct
                    cov = n / total if total else 0.0
                    candidates.append(make_candidate(
                        "model_horizon_regime", model, horizon,
                        f"model={model}, horizon={horizon}, {reg_col}={regime_val}",
                        n, cov, acc, correct, incorrect,
                        acc >= 0.60, n >= 100, True, True, "conditional",
                    ))

    return candidates


def audit_time(df: pd.DataFrame) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    total = len(df)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    df["year"] = df["date"].dt.year.astype(str)

    for model in sorted(df["model"].unique()):
        m_df = df[df["model"] == model]
        for horizon in sorted(m_df["horizon"].unique()):
            mh_df = m_df[m_df["horizon"] == horizon]

            for period_col, period_label in [("month", "month"), ("quarter", "quarter"), ("year", "year")]:
                for period_val in sorted(mh_df[period_col].dropna().unique()):
                    p_df = mh_df[mh_df[period_col] == period_val]
                    n = len(p_df)
                    if n < 50:
                        continue
                    acc = p_df["is_correct"].mean()
                    correct = int(p_df["is_correct"].sum())
                    incorrect = n - correct
                    cov = n / total if total else 0.0
                    candidates.append(make_candidate(
                        "model_horizon_time", model, horizon,
                        f"model={model}, horizon={horizon}, {period_label}={period_val}",
                        n, cov, acc, correct, incorrect,
                        acc >= 0.60, n >= 100, True, True, "exploratory",
                    ))

    return candidates


def audit_combined(df: pd.DataFrame) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    total = len(df)

    for model in sorted(df["model"].unique()):
        m_df = df[df["model"] == model]
        for horizon in sorted(m_df["horizon"].unique()):
            mh_df = m_df[m_df["horizon"] == horizon]
            parent_total = len(mh_df)

            for thresh in CONFIDENCE_THRESHOLDS:
                sel = mh_df[mh_df["confidence"] >= thresh]
                if len(sel) < 100:
                    continue

                for ticker in sorted(sel["ticker"].unique()):
                    t_df = sel[sel["ticker"] == ticker]
                    n = len(t_df)
                    if n < 50:
                        continue
                    acc = t_df["is_correct"].mean()
                    correct = int(t_df["is_correct"].sum())
                    incorrect = n - correct
                    cov = n / parent_total if parent_total else 0.0
                    candidates.append(make_candidate(
                        "model_horizon_conf_ticker", model, horizon,
                        f"model={model}, horizon={horizon}, conf>={thresh}, ticker={ticker}",
                        n, cov, acc, correct, incorrect,
                        acc >= 0.60, n >= 100, cov >= 0.10, True, "exploratory",
                    ))

    for reg_col in ["regime", "volatility_regime"]:
        if reg_col not in df.columns:
            continue
        for model in sorted(df["model"].unique()):
            m_df = df[df["model"] == model]
            for horizon in sorted(m_df["horizon"].unique()):
                mh_df = m_df[m_df["horizon"] == horizon]
                parent_total = len(mh_df)

                for thresh in CONFIDENCE_THRESHOLDS:
                    sel = mh_df[mh_df["confidence"] >= thresh]
                    if len(sel) < 100:
                        continue
                    for regime_val in sorted(sel[reg_col].dropna().unique()):
                        r_df = sel[sel[reg_col] == regime_val]
                        n = len(r_df)
                        if n < 50:
                            continue
                        acc = r_df["is_correct"].mean()
                        correct = int(r_df["is_correct"].sum())
                        incorrect = n - correct
                        cov = n / parent_total if parent_total else 0.0
                        candidates.append(make_candidate(
                            "model_horizon_conf_regime", model, horizon,
                            f"model={model}, horizon={horizon}, conf>={thresh}, {reg_col}={regime_val}",
                            n, cov, acc, correct, incorrect,
                            acc >= 0.60, n >= 100, cov >= 0.10, True, "exploratory",
                        ))

    return candidates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    passing = [c for c in candidates if c["pass_60pct"] == "yes"]
    passing_sorted = sorted(passing, key=lambda x: float(x["accuracy"]), reverse=True)

    lines = [
        "# VN30 Hourly 2015 - All >60% Candidates",
        "",
        f"- Generated at UTC: `{now_utc()}`.",
        f"- Total candidates evaluated: {len(candidates)}.",
        f"- Candidates passing 60%: {len(passing)}.",
        "- All results are from the existing benchmark artifact only.",
        "",
        "## Top Passing Candidates (by accuracy)",
        "",
        markdown_table(CANDIDATE_COLUMNS, passing_sorted, max_rows=50),
        "",
        "## All Passing Candidates",
        "",
        markdown_table(CANDIDATE_COLUMNS, passing_sorted),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    passing = [c for c in candidates if c["pass_60pct"] == "yes"]
    passing_sorted = sorted(passing, key=lambda x: float(x["accuracy"]), reverse=True)

    global_pass = [c for c in passing if c["candidate_type"] in ("global_model", "global_model_horizon")]
    conf_pass = [c for c in passing if c["candidate_type"] == "model_horizon_confidence"]
    ticker_pass = [c for c in passing if c["candidate_type"] == "model_horizon_ticker"]
    regime_pass = [c for c in passing if c["candidate_type"] == "model_horizon_regime"]
    time_pass = [c for c in passing if c["candidate_type"] == "model_horizon_time"]
    combined_pass = [c for c in passing if c["candidate_type"] in ("model_horizon_conf_ticker", "model_horizon_conf_regime")]

    safe_claims = [c for c in passing if c["allowed_claim_level"] == "global"]
    conditional_claims = [c for c in passing if c["allowed_claim_level"] == "conditional"]
    exploratory_claims = [c for c in passing if c["allowed_claim_level"] == "exploratory"]

    best_global = global_pass[0] if global_pass else None
    best_conf_qual = None
    for c in conf_pass:
        if c["min_row_floor_met"] == "yes" and c["coverage_floor_met"] == "yes":
            if float(c["accuracy"]) >= 0.60:
                best_conf_qual = c
                break

    lines = [
        "# VN30 Hourly 2015 - Above 60% Audit Summary",
        "",
        f"- Generated at UTC: `{now_utc()}`.",
        f"- Source: `{rel(PREDICTIONS_PATH)}`.",
        f"- Total predictions in benchmark: 77,692.",
        f"- Global accuracy: 51.34%.",
        "",
        "## Are there any global model/horizon rows above 60%?",
        "",
    ]
    if best_global:
        lines.extend([
            f"**YES.** {best_global['model']} h={best_global['horizon']} achieved {fmt_pct(best_global['accuracy'])} accuracy.",
            "",
        ])
    else:
        lines.extend([
            "**NO.** No global model or model/horizon combination exceeds 60% accuracy.",
            f"- Best global candidates: {len(global_pass)} (all are conditional or exploratory).",
            "",
        ])

    lines.extend([
        "## Are there any coverage-qualified confidence slices above 60%?",
        "",
    ])
    if best_conf_qual:
        lines.extend([
            f"**YES.** {best_conf_qual['model']} h={best_conf_qual['horizon']} at conf>={best_conf_qual['filter_description'].split('conf>=')[1].split(',')[0]} achieved {fmt_pct(best_conf_qual['accuracy'])} with coverage {fmt_pct(best_conf_qual['coverage_ratio'])}.",
            "",
        ])
    else:
        lines.extend([
            "**NO.** No confidence-filtered slice meets both the 60% accuracy threshold AND the coverage/row count floors simultaneously.",
            "",
        ])

    lines.extend([
        "## Are there any ticker-level rows above 60%?",
        "",
    ])
    if ticker_pass:
        best_ticker = ticker_pass[0]
        lines.extend([
            f"**YES.** {len(ticker_pass)} ticker-level candidates pass 60%.",
            f"- Best: {best_ticker['filter_description']} at {fmt_pct(best_ticker['accuracy'])} ({best_ticker['observations']} obs).",
            "- These are exploratory/post-hoc and cannot be claimed as global results.",
            "",
        ])
    else:
        lines.extend([
            "**NO.** No ticker-level rows exceed 60% with sufficient observations.",
            "",
        ])

    lines.extend([
        "## Are there any regime rows above 60%?",
        "",
    ])
    if regime_pass:
        best_regime = regime_pass[0]
        lines.extend([
            f"**YES.** {len(regime_pass)} regime-level candidates pass 60%.",
            f"- Best: {best_regime['filter_description']} at {fmt_pct(best_regime['accuracy'])} ({best_regime['observations']} obs).",
            "- These are conditional/diagnostic and cannot be claimed as global results.",
            "",
        ])
    else:
        lines.extend([
            "**NO.** No regime-level rows exceed 60%.",
            "",
        ])

    lines.extend([
        "## Which results are claim-safe and which are only exploratory?",
        "",
        f"- **Safe global claims (>60%):** {len(safe_claims)}.",
        f"- **Conditional claims (>60% with coverage disclosure):** {len(conditional_claims)}.",
        f"- **Exploratory/post-hoc observations (>60%):** {len(exploratory_claims)}.",
        "",
    ])

    if safe_claims:
        lines.append("### Safe Global Claims")
        lines.append("")
        lines.append(markdown_table(CANDIDATE_COLUMNS, safe_claims))
        lines.append("")

    if conditional_claims:
        lines.append("### Conditional Claims (require coverage/row disclosure)")
        lines.append("")
        lines.append(markdown_table(CANDIDATE_COLUMNS, sorted(conditional_claims, key=lambda x: float(x["accuracy"]), reverse=True), max_rows=20))
        lines.append("")

    if exploratory_claims:
        lines.append("### Exploratory/Post-Hoc Observations")
        lines.append("")
        lines.append(markdown_table(CANDIDATE_COLUMNS, sorted(exploratory_claims, key=lambda x: float(x["accuracy"]), reverse=True), max_rows=30))
        lines.append("")

    lines.extend([
        "## Top 20 Passing Candidates Overall",
        "",
        markdown_table(CANDIDATE_COLUMNS, passing_sorted, max_rows=20),
        "",
        "## Boundary",
        "",
        "- No trading-readiness, profitability, or live deployment claim is made.",
        "- All confidence-filtered and combined results are post-hoc diagnostics.",
        "- Global claims require all 30 active tickers and full evaluation coverage.",
        "- No new data was fetched. No prediction labels were edited.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    print("Loading predictions...")
    df = load_predictions(PREDICTIONS_PATH)
    print(f"  Loaded {len(df)} predictions.")

    all_candidates: list[dict[str, Any]] = []

    print("Auditing global slices...")
    all_candidates.extend(audit_global(df))
    print(f"  Global candidates: {len([c for c in all_candidates if c['candidate_type'].startswith('global')])}")

    print("Auditing ticker slices...")
    ticker_candidates = audit_ticker(df)
    all_candidates.extend(ticker_candidates)
    print(f"  Ticker candidates: {len(ticker_candidates)}")

    print("Auditing confidence slices...")
    conf_candidates = audit_confidence(df)
    all_candidates.extend(conf_candidates)
    print(f"  Confidence candidates: {len(conf_candidates)}")

    print("Auditing regime slices...")
    regime_candidates = audit_regime(df)
    all_candidates.extend(regime_candidates)
    print(f"  Regime candidates: {len(regime_candidates)}")

    print("Auditing time slices...")
    time_candidates = audit_time(df)
    all_candidates.extend(time_candidates)
    print(f"  Time candidates: {len(time_candidates)}")

    print("Auditing combined slices...")
    combined_candidates = audit_combined(df)
    all_candidates.extend(combined_candidates)
    print(f"  Combined candidates: {len(combined_candidates)}")

    print(f"\nTotal candidates: {len(all_candidates)}")
    passing = [c for c in all_candidates if c["pass_60pct"] == "yes"]
    print(f"Passing 60%: {len(passing)}")

    print(f"\nWriting CSV to {rel(CSV_PATH)}...")
    write_csv(CSV_PATH, all_candidates)

    print(f"Writing MD to {rel(MD_PATH)}...")
    write_md(MD_PATH, all_candidates)

    print(f"Writing summary to {rel(SUMMARY_PATH)}...")
    write_summary(SUMMARY_PATH, all_candidates)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())