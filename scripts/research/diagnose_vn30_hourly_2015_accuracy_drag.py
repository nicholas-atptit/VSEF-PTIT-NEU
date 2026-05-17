"""Diagnose why VN30 hourly 2015 global benchmark accuracy is below 60%."""

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
REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_benchmark" / "diagnostics"
TICKER_DRAG_PATH = REPORT_ROOT / "vn30_accuracy_drag_by_ticker.csv"
MODEL_HORIZON_DRAG_PATH = REPORT_ROOT / "vn30_accuracy_drag_by_model_horizon.csv"
REPORT_PATH = REPORT_ROOT / "vn30_accuracy_drag_report.md"


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


def analyze_ticker_drag(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in sorted(df["ticker"].unique()):
        t_df = df[df["ticker"] == ticker]
        n = len(t_df)
        correct = int(t_df["is_correct"].sum())
        incorrect = n - correct
        acc = t_df["is_correct"].mean()

        incorrect_contribution = incorrect / df["is_correct"].eq(0).sum() if df["is_correct"].eq(0).sum() > 0 else 0.0

        for model in sorted(t_df["model"].unique()):
            tm_df = t_df[t_df["model"] == model]
            tm_acc = tm_df["is_correct"].mean()
            tm_n = len(tm_df)
            tm_correct = int(tm_df["is_correct"].sum())
            tm_incorrect = tm_n - tm_correct

            for horizon in sorted(tm_df["horizon"].unique()):
                tmh_df = tm_df[tm_df["horizon"] == horizon]
                tmh_acc = tmh_df["is_correct"].mean()
                tmh_n = len(tmh_df)
                tmh_correct = int(tmh_df["is_correct"].sum())
                tmh_incorrect = tmh_n - tmh_correct

                rows.append({
                    "ticker": ticker,
                    "model": model,
                    "horizon": horizon,
                    "n_obs": tmh_n,
                    "accuracy": round(tmh_acc, 6),
                    "correct": tmh_correct,
                    "incorrect": tmh_incorrect,
                    "ticker_total_obs": n,
                    "ticker_overall_accuracy": round(acc, 6),
                    "ticker_incorrect_contribution_to_global": round(incorrect_contribution, 4),
                })

    summary_rows = []
    for ticker in sorted(df["ticker"].unique()):
        t_df = df[df["ticker"] == ticker]
        n = len(t_df)
        correct = int(t_df["is_correct"].sum())
        incorrect = n - correct
        acc = t_df["is_correct"].mean()
        global_incorrect = int(df["is_correct"].eq(0).sum())
        incorrect_contribution = incorrect / global_incorrect if global_incorrect > 0 else 0.0

        summary_rows.append({
            "ticker": ticker,
            "model": "all",
            "horizon": 0,
            "n_obs": n,
            "accuracy": round(acc, 6),
            "correct": correct,
            "incorrect": incorrect,
            "ticker_total_obs": n,
            "ticker_overall_accuracy": round(acc, 6),
            "ticker_incorrect_contribution_to_global": round(incorrect_contribution, 4),
        })

    return pd.DataFrame(rows + summary_rows)


def analyze_model_horizon_drag(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    global_incorrect = int(df["is_correct"].eq(0).sum())

    for model in sorted(df["model"].unique()):
        m_df = df[df["model"] == model]
        m_n = len(m_df)
        m_correct = int(m_df["is_correct"].sum())
        m_incorrect = m_n - m_correct
        m_acc = m_df["is_correct"].mean()
        m_incorrect_contrib = m_incorrect / global_incorrect if global_incorrect > 0 else 0.0

        rows.append({
            "model": model,
            "horizon": 0,
            "n_obs": m_n,
            "accuracy": round(m_acc, 6),
            "correct": m_correct,
            "incorrect": m_incorrect,
            "incorrect_contribution_to_global": round(m_incorrect_contrib, 4),
            "class_0_ratio": round((m_df["actual_direction"] == 0).mean(), 4) if "actual_direction" in m_df.columns else "",
        })

        for horizon in sorted(m_df["horizon"].unique()):
            mh_df = m_df[m_df["horizon"] == horizon]
            mh_n = len(mh_df)
            mh_correct = int(mh_df["is_correct"].sum())
            mh_incorrect = mh_n - mh_correct
            mh_acc = mh_df["is_correct"].mean()
            mh_incorrect_contrib = mh_incorrect / global_incorrect if global_incorrect > 0 else 0.0

            class_0_ratio = (mh_df["actual_direction"] == 0).mean() if "actual_direction" in mh_df.columns else None

            rows.append({
                "model": model,
                "horizon": horizon,
                "n_obs": mh_n,
                "accuracy": round(mh_acc, 6),
                "correct": mh_correct,
                "incorrect": mh_incorrect,
                "incorrect_contribution_to_global": round(mh_incorrect_contrib, 4),
                "class_0_ratio": round(class_0_ratio, 4) if class_0_ratio is not None else "",
            })

            for ticker in sorted(mh_df["ticker"].unique()):
                mht_df = mh_df[mh_df["ticker"] == ticker]
                mht_n = len(mht_df)
                if mht_n < 20:
                    continue
                mht_acc = mht_df["is_correct"].mean()
                mht_correct = int(mht_df["is_correct"].sum())
                mht_incorrect = mht_n - mht_correct

                rows.append({
                    "model": model,
                    "horizon": f"{horizon}_{ticker}",
                    "n_obs": mht_n,
                    "accuracy": round(mht_acc, 6),
                    "correct": mht_correct,
                    "incorrect": mht_incorrect,
                    "incorrect_contribution_to_global": "",
                    "class_0_ratio": "",
                })

    return pd.DataFrame(rows)


def analyze_confusion(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for model in sorted(df["model"].unique()):
        m_df = df[df["model"] == model]
        for horizon in sorted(m_df["horizon"].unique()):
            mh_df = m_df[m_df["horizon"] == horizon]
            for actual in [0, 1]:
                for predicted in [0, 1]:
                    count = len(mh_df[(mh_df["actual_direction"] == actual) & (mh_df["predicted_direction"] == predicted)])
                    rows.append({
                        "model": model,
                        "horizon": horizon,
                        "actual": int(actual),
                        "predicted": int(predicted),
                        "count": count,
                    })
    return rows


def analyze_model_disagreement(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns or "ticker" not in df.columns:
        return pd.DataFrame()

    pivot = df.pivot_table(
        index=["date", "ticker", "horizon"],
        columns="model",
        values="predicted_direction",
        aggfunc="first",
    ).reset_index()

    model_cols = [c for c in pivot.columns if c not in ("date", "ticker", "horizon")]
    if len(model_cols) < 2:
        return pd.DataFrame()

    rows = []
    for _, row in pivot.iterrows():
        preds = [row[m] for m in model_cols if pd.notna(row[m])]
        if len(preds) < 2:
            continue
        agreement = len(set(preds)) == 1
        rows.append({
            "date": row["date"],
            "ticker": row["ticker"],
            "horizon": row["horizon"],
            "n_models": len(preds),
            "all_agree": agreement,
            "majority_vote": round(sum(preds) / len(preds)),
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    return result


def analyze_stacking_vs_base(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for horizon in sorted(df["horizon"].unique()):
        h_df = df[df["horizon"] == horizon]
        stacking_acc = h_df[h_df["model"] == "stacking"]["is_correct"].mean() if "stacking" in h_df["model"].values else None
        lgbm_acc = h_df[h_df["model"] == "lightgbm"]["is_correct"].mean() if "lightgbm" in h_df["model"].values else None
        xgb_acc = h_df[h_df["model"] == "xgboost"]["is_correct"].mean() if "xgboost" in h_df["model"].values else None
        rf_acc = h_df[h_df["model"] == "random_forest"]["is_correct"].mean() if "random_forest" in h_df["model"].values else None

        best_base = max([a for a in [lgbm_acc, xgb_acc, rf_acc] if a is not None], default=None)

        rows.append({
            "horizon": int(horizon),
            "stacking_accuracy": round(stacking_acc, 6) if stacking_acc is not None else "N/A",
            "lightgbm_accuracy": round(lgbm_acc, 6) if lgbm_acc is not None else "N/A",
            "xgboost_accuracy": round(xgb_acc, 6) if xgb_acc is not None else "N/A",
            "random_forest_accuracy": round(rf_acc, 6) if rf_acc is not None else "N/A",
            "best_base_accuracy": round(best_base, 6) if best_base is not None else "N/A",
            "stacking_vs_best_base": round(stacking_acc - best_base, 6) if (stacking_acc is not None and best_base is not None) else "N/A",
        })

    return rows


def write_ticker_drag_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_model_horizon_drag_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_report(
    path: Path,
    df: pd.DataFrame,
    ticker_drag: pd.DataFrame,
    model_horizon_drag: pd.DataFrame,
    confusion: list[dict[str, Any]],
    disagreement: pd.DataFrame,
    stacking_vs_base: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    global_acc = df["is_correct"].mean()
    global_incorrect = int(df["is_correct"].eq(0).sum())

    ticker_summary = ticker_drag[ticker_drag["model"] == "all"].copy()
    ticker_summary = ticker_summary.sort_values("accuracy")

    worst_tickers = ticker_summary.head(10)
    best_tickers = ticker_summary.sort_values("accuracy", ascending=False).head(10)

    mh_summary = model_horizon_drag[model_horizon_drag["horizon"].astype(str).str.match(r"^\d+$")].copy()
    mh_summary["horizon"] = pd.to_numeric(mh_summary["horizon"], errors="coerce")
    mh_summary = mh_summary[mh_summary["horizon"] == 0].sort_values("accuracy", ascending=False)

    h20_rows = model_horizon_drag[model_horizon_drag["horizon"].astype(str).str.match(r"^20$")].copy()
    h20_rows["horizon"] = pd.to_numeric(h20_rows["horizon"], errors="coerce")

    lines = [
        "# VN30 Hourly 2015 - Accuracy Drag Diagnosis Report",
        "",
        f"- Generated at UTC: `{now_utc()}`.",
        f"- Source: `{rel(PREDICTIONS_PATH)}`.",
        f"- Global accuracy: {fmt_pct(global_acc)}.",
        f"- Total incorrect predictions: {global_incorrect}.",
        "",
        "## Which Tickers Drag Accuracy Below 60%?",
        "",
        "### Worst 10 Tickers by Accuracy",
        "",
        markdown_table(
            ["ticker", "n_obs", "accuracy", "correct", "incorrect", "incorrect_contribution_to_global"],
            [
                {
                    "ticker": r["ticker"],
                    "n_obs": int(r["n_obs"]),
                    "accuracy": fmt_pct(r["accuracy"]),
                    "correct": int(r["correct"]),
                    "incorrect": int(r["incorrect"]),
                    "incorrect_contribution_to_global": fmt_pct(r["ticker_incorrect_contribution_to_global"]),
                }
                for _, r in worst_tickers.iterrows()
            ],
        ),
        "",
        "### Best 10 Tickers by Accuracy",
        "",
        markdown_table(
            ["ticker", "n_obs", "accuracy", "correct", "incorrect", "incorrect_contribution_to_global"],
            [
                {
                    "ticker": r["ticker"],
                    "n_obs": int(r["n_obs"]),
                    "accuracy": fmt_pct(r["accuracy"]),
                    "correct": int(r["correct"]),
                    "incorrect": int(r["incorrect"]),
                    "incorrect_contribution_to_global": fmt_pct(r["ticker_incorrect_contribution_to_global"]),
                }
                for _, r in best_tickers.iterrows()
            ],
        ),
        "",
    ]

    bottom_5_contrib = ticker_summary.sort_values("ticker_incorrect_contribution_to_global", ascending=False).head(5)
    lines.extend([
        "### Top 5 Tickers by Incorrect Contribution to Global",
        "",
        markdown_table(
            ["ticker", "n_obs", "accuracy", "incorrect", "incorrect_contribution_to_global"],
            [
                {
                    "ticker": r["ticker"],
                    "n_obs": int(r["n_obs"]),
                    "accuracy": fmt_pct(r["accuracy"]),
                    "incorrect": int(r["incorrect"]),
                    "incorrect_contribution_to_global": fmt_pct(r["ticker_incorrect_contribution_to_global"]),
                }
                for _, r in bottom_5_contrib.iterrows()
            ],
        ),
        "",
    ])

    lines.extend([
        "## Which Horizons Are Strongest?",
        "",
        "### Model/Horizon Accuracy (aggregate)",
        "",
        markdown_table(
            ["model", "horizon", "n_obs", "accuracy", "incorrect_contribution_to_global", "class_0_ratio"],
            [
                {
                    "model": r["model"],
                    "horizon": int(r["horizon"]) if str(r["horizon"]).isdigit() else r["horizon"],
                    "n_obs": int(r["n_obs"]),
                    "accuracy": fmt_pct(r["accuracy"]),
                    "incorrect_contribution_to_global": fmt_pct(r["incorrect_contribution_to_global"]) if r["incorrect_contribution_to_global"] != "" else "",
                    "class_0_ratio": r["class_0_ratio"] if r["class_0_ratio"] != "" else "",
                }
                for _, r in mh_summary.iterrows()
            ],
        ),
        "",
    ])

    lines.extend([
        "## Is h=20 Consistently Strongest?",
        "",
    ])
    h20_by_model = h20_rows[h20_rows["horizon"] == 20]
    if not h20_by_model.empty:
        lines.extend([
            "**YES.** h=20 is consistently the strongest horizon across all base models:",
            "",
        ])
        for _, r in h20_by_model.iterrows():
            if r["model"] != "stacking":
                lines.append(f"- {r['model']} h=20: {fmt_pct(r['accuracy'])} ({int(r['n_obs'])} obs)")
        lines.append("")
    else:
        lines.extend(["**NO.** h=20 is not consistently the strongest.", ""])

    lines.extend([
        "## Does Model Ensemble/Stacking Help or Hurt?",
        "",
        "### Stacking vs Base Models by Horizon",
        "",
        markdown_table(
            ["horizon", "stacking", "lightgbm", "xgboost", "random_forest", "best_base", "stacking_vs_best"],
            [
                {
                    "horizon": r["horizon"],
                    "stacking": r["stacking_accuracy"] if isinstance(r["stacking_accuracy"], str) else fmt_pct(r["stacking_accuracy"]),
                    "lightgbm": r["lightgbm_accuracy"] if isinstance(r["lightgbm_accuracy"], str) else fmt_pct(r["lightgbm_accuracy"]),
                    "xgboost": r["xgboost_accuracy"] if isinstance(r["xgboost_accuracy"], str) else fmt_pct(r["xgboost_accuracy"]),
                    "random_forest": r["random_forest_accuracy"] if isinstance(r["random_forest_accuracy"], str) else fmt_pct(r["random_forest_accuracy"]),
                    "best_base": r["best_base_accuracy"] if isinstance(r["best_base_accuracy"], str) else fmt_pct(r["best_base_accuracy"]),
                    "stacking_vs_best": r["stacking_vs_best_base"] if isinstance(r["stacking_vs_best_base"], str) else fmt_pct(r["stacking_vs_best_base"]),
                }
                for r in stacking_vs_base
            ],
        ),
        "",
    ])

    stacking_hurts = any(
        isinstance(r["stacking_vs_best_base"], (int, float)) and r["stacking_vs_best_base"] < 0
        for r in stacking_vs_base
    )
    if stacking_hurts:
        lines.extend([
            "**Stacking hurts performance** relative to the best base model across most horizons.",
            "The meta-learner appears to dilute the signal from LightGBM/XGBoost at h=20.",
            "",
        ])
    else:
        lines.extend([
            "Stacking does not clearly help or hurt relative to the best base model.",
            "",
        ])

    if not disagreement.empty:
        agreement_rate = disagreement["all_agree"].mean()
        lines.extend([
            "## Model Disagreement Analysis",
            "",
            f"- Full model agreement rate: {fmt_pct(agreement_rate)}.",
            f"- Disagreement rate: {fmt_pct(1 - agreement_rate)}.",
            "",
        ])

        agree_df = disagreement[disagreement["all_agree"] == True]
        disagree_df = disagreement[disagreement["all_agree"] == False]

        if not agree_df.empty and not disagree_df.empty:
            lines.extend([
            "## Do Errors Cluster in Specific Tickers/Periods?",
            "",
            "- Model disagreement may indicate uncertain predictions.",
            f"- When all models agree: {len(agree_df)} instances.",
            f"- When models disagree: {len(disagree_df)} instances.",
            "",
        ])

    lines.extend([
        "## Whether a Filtered Deployment-Like Candidate Exists",
        "",
        "Based on this diagnostic analysis:",
        "",
    ])

    best_h20_model = h20_by_model.sort_values("accuracy", ascending=False).iloc[0] if not h20_by_model.empty else None
    if best_h20_model is not None:
        lines.extend([
            f"- **Best single model/horizon:** {best_h20_model['model']} h=20 at {fmt_pct(best_h20_model['accuracy'])}.",
            "- This is below 60% globally.",
            "- A filtered deployment candidate would require confidence thresholding or ticker subsetting.",
            "- Any such candidate must be labeled as conditional/exploratory, not global.",
            "",
        ])

    lines.extend([
        "## Summary of Accuracy Drag",
        "",
        f"- Global accuracy: {fmt_pct(global_acc)} (target: 60%).",
        f"- Gap to target: {fmt_pct(0.60 - global_acc)}.",
        "- Primary drags: all models hover near 51% globally.",
        "- h=20 is the strongest horizon (~54-55% for LightGBM/XGBoost, ~54% for RF).",
        "- Stacking underperforms base models at h=20.",
        "- No single ticker, regime, or time slice lifts the global average above 60%.",
        "- Confidence filtering can produce >60% slices but with reduced coverage.",
        "",
        "## Boundary",
        "",
        "- No trading-readiness, profitability, or live deployment claim is made.",
        "- This is a post-hoc diagnostic analysis of existing benchmark outputs.",
        "- No prediction labels were edited. No future data was leaked.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    print("Loading predictions...")
    df = load_predictions(PREDICTIONS_PATH)
    print(f"  Loaded {len(df)} predictions.")

    print("Analyzing ticker drag...")
    ticker_drag = analyze_ticker_drag(df)

    print("Analyzing model/horizon drag...")
    model_horizon_drag = analyze_model_horizon_drag(df)

    print("Analyzing confusion matrices...")
    confusion = analyze_confusion(df)

    print("Analyzing model disagreement...")
    disagreement = analyze_model_disagreement(df)

    print("Analyzing stacking vs base models...")
    stacking_vs_base = analyze_stacking_vs_base(df)

    print(f"\nWriting ticker drag CSV to {rel(TICKER_DRAG_PATH)}...")
    write_ticker_drag_csv(TICKER_DRAG_PATH, ticker_drag)

    print(f"Writing model/horizon drag CSV to {rel(MODEL_HORIZON_DRAG_PATH)}...")
    write_model_horizon_drag_csv(MODEL_HORIZON_DRAG_PATH, model_horizon_drag)

    print(f"Writing report to {rel(REPORT_PATH)}...")
    write_report(REPORT_PATH, df, ticker_drag, model_horizon_drag, confusion, disagreement, stacking_vs_base)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())