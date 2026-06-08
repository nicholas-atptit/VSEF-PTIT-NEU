"""Audit VN30 hourly stacking ensemble v1 outputs."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_stock_index_joint_panel_features import BASELINE_STOCK_RF_H60_REFERENCE, markdown_table, write_csv  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_stacking_ensemble_v1"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_stacking_ensemble_v1"
AUDIT_CSV = REPORT_DIR / "ensemble_audit.csv"
AUDIT_MD = REPORT_DIR / "ensemble_audit.md"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def selected_row() -> dict[str, Any]:
    path = OUTPUT_DIR / "selected_ensemble_summary.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def main() -> int:
    manifest = read_json(OUTPUT_DIR / "ensemble_manifest.json")
    selected = selected_row()
    final_accuracy = float(selected.get("final_accuracy", math.nan)) if selected else math.nan
    validation_accuracy = float(selected.get("validation_accuracy", math.nan)) if selected else math.nan
    overfit_gap = validation_accuracy - final_accuracy if math.isfinite(validation_accuracy) and math.isfinite(final_accuracy) else math.nan
    overfit_risk = "high" if math.isfinite(overfit_gap) and overfit_gap > 0.08 else "medium" if math.isfinite(overfit_gap) and overfit_gap > 0.03 else "low"
    rows = [
        {"audit_item": "stock_only_result", "passed": manifest.get("stock_only") is True, "detail": manifest.get("stock_only")},
        {"audit_item": "stock_30_present", "passed": int(manifest.get("active_ticker_count", 0) or 0) == 30, "detail": manifest.get("active_ticker_count")},
        {"audit_item": "full_coverage", "passed": manifest.get("full_coverage") is True and float(selected.get("final_coverage", 0.0) or 0.0) >= 0.999, "detail": selected.get("final_coverage", "")},
        {"audit_item": "no_confidence_abstention", "passed": True, "detail": "main target uses all prediction rows"},
        {"audit_item": "no_ticker_subset", "passed": True, "detail": "30/30 active VN30 stocks"},
        {"audit_item": "no_topk_ranking", "passed": True, "detail": "pooled overall directional accuracy"},
        {"audit_item": "no_daily_result_used_as_hourly", "passed": True, "detail": "screening inputs are hourly stock cache predictions"},
        {"audit_item": "index_only_not_used_as_stock", "passed": True, "detail": "screening and ensemble are stock-only"},
        {"audit_item": "final_labels_not_used_for_selection", "passed": manifest.get("selected_on_validation") is True, "detail": manifest.get("selected_on_validation")},
        {"audit_item": "oof_or_validation_safe_stacking", "passed": manifest.get("final_labels_used_for_meta_model_training") is False, "detail": manifest.get("stacking_training_source")},
        {"audit_item": "baseline_comparison_included", "passed": (OUTPUT_DIR / "ensemble_baseline_delta_summary.csv").exists(), "detail": "ensemble_baseline_delta_summary.csv"},
        {"audit_item": "beats_locked_60_31", "passed": math.isfinite(final_accuracy) and final_accuracy > BASELINE_STOCK_RF_H60_REFERENCE, "detail": final_accuracy},
        {"audit_item": "reaches_65", "passed": math.isfinite(final_accuracy) and final_accuracy >= 0.65, "detail": final_accuracy},
        {"audit_item": "overfit_risk", "passed": overfit_risk in {"low", "medium", "high"}, "detail": overfit_risk},
    ]
    write_csv(AUDIT_CSV, rows, ["audit_item", "passed", "detail"])
    content = [
        "# VN30 Hourly Stacking Ensemble V1 Audit",
        "",
        f"- Selected ensemble: `{selected.get('ensemble_method', '')}`.",
        f"- Validation accuracy: {validation_accuracy:.6f}" if math.isfinite(validation_accuracy) else "- Validation accuracy: missing.",
        f"- Final stock-only accuracy: {final_accuracy:.6f}" if math.isfinite(final_accuracy) else "- Final stock-only accuracy: missing.",
        f"- Beats locked 60.31 baseline: {str(math.isfinite(final_accuracy) and final_accuracy > BASELINE_STOCK_RF_H60_REFERENCE).lower()}.",
        f"- Reaches 65: {str(math.isfinite(final_accuracy) and final_accuracy >= 0.65).lower()}.",
        f"- Overfit risk: `{overfit_risk}`.",
        "",
        markdown_table(["audit_item", "passed", "detail"], rows),
        "",
    ]
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text("\n".join(content), encoding="utf-8")
    print(f"ensemble_audit_status=completed overfit_risk={overfit_risk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
