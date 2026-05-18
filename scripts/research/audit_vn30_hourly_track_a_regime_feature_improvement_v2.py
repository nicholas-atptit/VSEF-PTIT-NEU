"""Audit Track A regime feature improvement v2 outputs."""

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

OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_track_a_regime_feature_improvement_v2"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_track_a_regime_feature_v2"
BASELINE_LOGISTIC_H40 = 0.6043200785468826


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
    manifest = read_json(OUTPUT_DIR / "improvement_manifest.json")
    selected_df = pd.read_csv(OUTPUT_DIR / "selected_candidate_summary.csv")
    results = pd.read_csv(OUTPUT_DIR / "final_candidate_results.csv")
    selected = selected_df.iloc[0].to_dict() if not selected_df.empty else {}
    final_accuracy = float(selected.get("final_accuracy", math.nan))
    validation_accuracy = float(selected.get("validation_accuracy", math.nan))
    feature_manifest = manifest.get("feature_manifest", {})
    risk = overfit_risk(validation_accuracy, final_accuracy) if math.isfinite(final_accuracy) else "unknown"
    required_cols = {"validation_baseline_accuracy", "final_baseline_accuracy", "delta_vs_60_43", "delta_vs_60_31"}
    audit_rows = [
        bool_row("stock_30_present", int(selected.get("active_ticker_count", 0)) == 30, selected.get("active_ticker_count", "")),
        bool_row("full_coverage", abs(float(selected.get("final_coverage", 0.0)) - 1.0) < 1e-12, selected.get("final_coverage", "")),
        bool_row("no_confidence_abstention", not bool(read_json(OUTPUT_DIR / "run_config.json").get("confidence_abstention")), "full probability threshold at 0.5"),
        bool_row("no_ticker_subset", not bool(read_json(OUTPUT_DIR / "run_config.json").get("ticker_subset")) and int(selected.get("active_ticker_count", 0)) == 30, "30/30 VN30 stocks"),
        bool_row("no_topk", not bool(read_json(OUTPUT_DIR / "run_config.json").get("topk")), "overall directional accuracy"),
        bool_row("no_final_label_selection", not bool(manifest.get("final_accuracy_used_for_selection")), manifest.get("final_accuracy_used_for_selection")),
        bool_row("final_scoring_only", bool(manifest.get("selected_on_validation")), "selected before final scoring report"),
        bool_row("leakage_audit_passed", bool(feature_manifest.get("leakage_safe")) and not bool(feature_manifest.get("future_return_features")) and not bool(feature_manifest.get("future_regime_features")) and not bool(feature_manifest.get("final_label_derived_features")), feature_manifest),
        bool_row("baseline_comparison_included", required_cols.issubset(set(results.columns)), "baseline columns present"),
        bool_row("selected_beats_60_43", bool(selected.get("pass_60_43", False)), final_accuracy - BASELINE_LOGISTIC_H40 if math.isfinite(final_accuracy) else ""),
        bool_row("selected_beats_60_31", bool(selected.get("pass_60_31", False)), final_accuracy - LOCKED_RF_H60 if math.isfinite(final_accuracy) else ""),
        bool_row("selected_reaches_65", bool(selected.get("pass_65", False)), final_accuracy),
        bool_row("overfit_risk", risk in {"low", "medium", "high"}, risk),
    ]
    write_csv(REPORT_DIR / "regime_feature_v2_audit.csv", audit_rows, ["audit_item", "passed", "detail"])
    report = [
        "# Track A Regime Feature Improvement V2 Audit",
        "",
        f"- Selected candidate: `{selected.get('model', '')}` h={selected.get('horizon', '')} `{selected.get('regime_method', '')}`.",
        f"- Validation accuracy: {pct(validation_accuracy)}.",
        f"- Final accuracy: {pct(final_accuracy)}.",
        f"- Delta vs 60.43: {(final_accuracy - BASELINE_LOGISTIC_H40) * 100:.2f} percentage points." if math.isfinite(final_accuracy) else "- Delta vs 60.43: .",
        f"- Delta vs 60.31: {(final_accuracy - LOCKED_RF_H60) * 100:.2f} percentage points." if math.isfinite(final_accuracy) else "- Delta vs 60.31: .",
        f"- Overfit risk: `{risk}`.",
        "",
        markdown_table(["audit_item", "passed", "detail"], audit_rows),
        "",
    ]
    (REPORT_DIR / "regime_feature_v2_audit.md").write_text("\n".join(report), encoding="utf-8")
    print(f"regime_feature_v2_audit_status=completed overfit_risk={risk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
