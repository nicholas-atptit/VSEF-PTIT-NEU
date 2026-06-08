"""Select validation-only shortlist for VN30 hourly ensemble/stacking."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_stock_index_joint_panel_features import markdown_table, write_csv  # noqa: E402

SCREENING_DIR = REPO_ROOT / "outputs" / "vn30_hourly_expanded_model_pool_screening"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_expanded_model_pool"
SHORTLIST_CSV = REPORT_DIR / "ensemble_shortlist.csv"
SHORTLIST_MD = REPORT_DIR / "ensemble_shortlist.md"

SHORTLIST_COLUMNS = [
    "candidate_id",
    "model_family",
    "model_name",
    "horizon",
    "feature_set",
    "hyperparams_id",
    "validation_accuracy",
    "validation_baseline_accuracy",
    "validation_delta_vs_baseline",
    "validation_stability_score",
    "diversity_score",
    "selected_for_ensemble",
    "selection_reason",
]


def pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number * 100:.2f}%"


def candidate_id(row: pd.Series) -> str:
    return f"{row['model_name']}__h{int(row['horizon'])}__{row['feature_set']}"


def main() -> int:
    results_path = SCREENING_DIR / "validation_candidate_results.csv"
    predictions_path = SCREENING_DIR / "candidate_predictions.csv"
    if not results_path.exists() or not predictions_path.exists():
        raise FileNotFoundError("Screening outputs are missing; run expanded model pool screening first.")
    results = pd.read_csv(results_path)
    predictions = pd.read_csv(predictions_path, low_memory=False)
    results = results[results["model_family"].ne("Baseline")].copy()
    results["candidate_id"] = results.apply(candidate_id, axis=1)
    for column in ("validation_accuracy", "validation_delta_vs_baseline", "validation_stability_score"):
        results[column] = pd.to_numeric(results[column], errors="coerce")
    eligible = results[
        (results["validation_delta_vs_baseline"] > 0)
        & results["validation_accuracy"].notna()
        & results["validation_stability_score"].notna()
    ].copy()
    if eligible.empty:
        eligible = results.sort_values(["validation_accuracy", "validation_stability_score"], ascending=False).head(3).copy()

    validation_predictions = predictions[predictions["split"].eq("validation")].copy()
    validation_predictions["key"] = (
        validation_predictions["datetime"].astype(str)
        + "|"
        + validation_predictions["ticker"].astype(str)
        + "|"
        + validation_predictions["horizon"].astype(str)
    )
    wide = validation_predictions.pivot_table(index="key", columns="candidate_id", values="prediction", aggfunc="first")

    selected_rows: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    pool = eligible.sort_values(["validation_accuracy", "validation_delta_vs_baseline", "validation_stability_score"], ascending=False)
    for _idx, row in pool.iterrows():
        cid = row["candidate_id"]
        if cid not in wide.columns:
            continue
        if not selected_ids:
            diversity = 1.0
            reason = "top_validation_accuracy_with_positive_lift"
        else:
            disagreements = []
            for selected_id in selected_ids:
                paired = wide[[cid, selected_id]].dropna()
                if len(paired):
                    disagreements.append(float((paired[cid] != paired[selected_id]).mean()))
            diversity = float(max(disagreements)) if disagreements else 0.0
            reason = "positive_validation_lift_and_error_diversity"
        if len(selected_ids) < 2 or diversity >= 0.05 or len(selected_ids) < 5:
            out = row.to_dict()
            out["diversity_score"] = diversity
            out["selected_for_ensemble"] = True
            out["selection_reason"] = reason
            selected_rows.append(out)
            selected_ids.append(cid)
        if len(selected_ids) >= 6:
            break

    if len(selected_rows) < 2 and len(pool) >= 2:
        for _idx, row in pool.iterrows():
            if row["candidate_id"] in selected_ids:
                continue
            out = row.to_dict()
            out["diversity_score"] = 0.0
            out["selected_for_ensemble"] = True
            out["selection_reason"] = "fallback_second_validation_candidate"
            selected_rows.append(out)
            selected_ids.append(row["candidate_id"])
            break

    write_csv(SHORTLIST_CSV, selected_rows, SHORTLIST_COLUMNS)
    content_rows = [
        {
            "candidate_id": row.get("candidate_id", ""),
            "model_name": row.get("model_name", ""),
            "horizon": row.get("horizon", ""),
            "feature_set": row.get("feature_set", ""),
            "validation_accuracy": pct(row.get("validation_accuracy")),
            "validation_delta_vs_baseline": pct(row.get("validation_delta_vs_baseline")),
            "diversity_score": f"{float(row.get('diversity_score', 0.0)):.4f}",
            "selection_reason": row.get("selection_reason", ""),
        }
        for row in selected_rows
    ]
    md = [
        "# VN30 Hourly Ensemble Shortlist",
        "",
        "- Selection uses validation accuracy, positive validation lift over baseline, validation stability, and prediction diversity.",
        "- Final accuracy is not used for selection.",
        "- The shortlist intentionally excludes weak or redundant candidates.",
        "",
        markdown_table(["candidate_id", "model_name", "horizon", "feature_set", "validation_accuracy", "validation_delta_vs_baseline", "diversity_score", "selection_reason"], content_rows),
        "",
    ]
    SHORTLIST_MD.parent.mkdir(parents=True, exist_ok=True)
    SHORTLIST_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"ensemble_shortlist_count={len(selected_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

