"""Audit Track A true all-algorithm stacking outputs."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.vn30_hourly_dual_track_common import LOCKED_RF_H60, REPO_ROOT, markdown_table, pct, read_json, write_csv  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_track_a_all_algorithm_stacking"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_track_a_true_stacking_all_algorithms"
PREVIOUS_LOGISTIC_H40 = 0.6043200785468826


def bool_row(item: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"audit_item": item, "passed": bool(passed), "detail": detail}


def overfit_risk(validation_accuracy: float, final_accuracy: float) -> str:
    gap = validation_accuracy - final_accuracy
    if gap >= 0.08:
        return "high"
    if gap >= 0.04:
        return "medium"
    return "low"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    oof_manifest = read_json(OUTPUT_DIR / "oof_manifest.json")
    stacking_manifest = read_json(OUTPUT_DIR / "stacking_manifest.json")
    base_scores = pd.read_csv(OUTPUT_DIR / "base_model_scores.csv")
    selected_df = pd.read_csv(OUTPUT_DIR / "selected_stack_summary.csv")
    all_rows = pd.read_csv(OUTPUT_DIR / "stacking_final_results.csv")
    selected = selected_df.iloc[0].to_dict() if not selected_df.empty else {}
    final_accuracy = float(selected.get("final_accuracy", math.nan))
    validation_accuracy = float(selected.get("validation_accuracy", math.nan))
    base_families = set(base_scores["model_family"].astype(str))
    required_families = {"random_forest", "extra_trees", "decision_tree_cart", "xgboost", "lightgbm", "logistic_regression"}
    risk = overfit_risk(validation_accuracy, final_accuracy) if math.isfinite(final_accuracy) else "unknown"
    audit_rows = [
        bool_row("true_stacking_used", bool(stacking_manifest.get("true_stacking_used")), stacking_manifest.get("true_stacking_used")),
        bool_row("all_base_algorithm_predictions_generated", required_families.issubset(base_families), ",".join(sorted(base_families))),
        bool_row("oof_validation_predictions_used", bool(oof_manifest.get("validation_oof_predictions_used")) and bool(stacking_manifest.get("meta_models_train_on_validation_oof_predictions_only")), "validation/OOF probabilities used as stack features"),
        bool_row("final_labels_not_used_for_stacking", not bool(stacking_manifest.get("final_labels_used_for_training_or_selection")), stacking_manifest.get("final_labels_used_for_training_or_selection")),
        bool_row("stock_30_present", int(selected.get("active_ticker_count", 0)) == 30, selected.get("active_ticker_count", "")),
        bool_row("full_coverage", abs(float(selected.get("final_coverage", 0.0)) - 1.0) < 1e-12, selected.get("final_coverage", "")),
        bool_row("no_confidence_abstention", not bool(oof_manifest.get("confidence_abstention")), oof_manifest.get("confidence_abstention")),
        bool_row("no_ticker_subset", not bool(oof_manifest.get("ticker_subset")) and int(selected.get("active_ticker_count", 0)) == 30, "30/30 VN30 stocks"),
        bool_row("no_topk", not bool(oof_manifest.get("topk")), oof_manifest.get("topk")),
        bool_row("baseline_comparison_included", {"validation_baseline_accuracy", "final_baseline_accuracy", "delta_vs_60_31", "delta_vs_60_43"}.issubset(set(all_rows.columns)), "baseline columns present"),
        bool_row("pass_60", bool(selected.get("pass_60", False)), final_accuracy),
        bool_row("pass_60_31", bool(selected.get("pass_60_31", False)), final_accuracy - LOCKED_RF_H60 if math.isfinite(final_accuracy) else ""),
        bool_row("pass_60_43", bool(selected.get("pass_60_43", False)), final_accuracy - PREVIOUS_LOGISTIC_H40 if math.isfinite(final_accuracy) else ""),
        bool_row("pass_65", bool(selected.get("pass_65", False)), final_accuracy),
        bool_row("overfit_risk", risk in {"low", "medium", "high"}, risk),
    ]
    write_csv(REPORT_DIR / "true_stacking_audit.csv", audit_rows, ["audit_item", "passed", "detail"])
    report = [
        "# Track A True Stacking All Algorithms Audit",
        "",
        f"- Selected stack: `{selected.get('stacking_method', '')}` h={selected.get('horizon', '')}.",
        f"- Validation accuracy: {pct(validation_accuracy)}.",
        f"- Final accuracy: {pct(final_accuracy)}.",
        f"- Delta vs 60.31: {(final_accuracy - LOCKED_RF_H60) * 100:.2f} percentage points." if math.isfinite(final_accuracy) else "- Delta vs 60.31: .",
        f"- Delta vs 60.43: {(final_accuracy - PREVIOUS_LOGISTIC_H40) * 100:.2f} percentage points." if math.isfinite(final_accuracy) else "- Delta vs 60.43: .",
        f"- Overfit risk: `{risk}`.",
        "",
        markdown_table(["audit_item", "passed", "detail"], audit_rows),
        "",
    ]
    (REPORT_DIR / "true_stacking_audit.md").write_text("\n".join(report), encoding="utf-8")
    print(f"true_stacking_audit_status=completed overfit_risk={risk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
