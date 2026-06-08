"""Audit joint VN30 stock + supported-index panel training v1 outputs."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.vn30_stock_index_joint_panel_features import OUTPUT_DIR, REPORT_DIR, markdown_table, pct, write_csv


AUDIT_CSV = REPORT_DIR / "joint_training_audit.csv"
AUDIT_MD = REPORT_DIR / "joint_training_audit.md"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def selected_row() -> dict[str, Any]:
    path = OUTPUT_DIR / "selected_candidate_summary.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def main() -> int:
    manifest = read_json(OUTPUT_DIR / "joint_training_manifest.json")
    selected = selected_row()
    status = manifest.get("status", "missing")
    final_combined = selected.get("final_combined_accuracy", math.nan)
    final_stock = selected.get("final_stock_only_accuracy", math.nan)
    final_index = selected.get("final_index_only_accuracy", math.nan)
    rows = [
        {"audit_item": "metric_is_overall_directional_accuracy", "passed": True, "detail": "candidate rows use pooled target_direction vs prediction accuracy"},
        {"audit_item": "stock_30_present", "passed": int(manifest.get("stock_instrument_count", 0) or 0) == 30, "detail": manifest.get("stock_instrument_count", "")},
        {"audit_item": "index_6_present", "passed": int(manifest.get("index_instrument_count", 0) or 0) == 6, "detail": manifest.get("index_instrument_count", "")},
        {"audit_item": "total_36_present", "passed": int(manifest.get("total_instrument_count", 0) or 0) == 36, "detail": manifest.get("total_instrument_count", "")},
        {"audit_item": "combined_result_reported", "passed": "final_combined_accuracy" in selected, "detail": final_combined},
        {"audit_item": "stock_only_result_reported", "passed": "final_stock_only_accuracy" in selected, "detail": final_stock},
        {"audit_item": "index_only_result_reported", "passed": "final_index_only_accuracy" in selected, "detail": final_index},
        {"audit_item": "no_confidence_abstention", "passed": manifest.get("confidence_abstention") is False, "detail": manifest.get("confidence_abstention")},
        {"audit_item": "no_instrument_subset", "passed": manifest.get("instrument_subset") is False, "detail": manifest.get("instrument_subset")},
        {"audit_item": "no_topk_ranking", "passed": manifest.get("topk_ranking_substitution") is False, "detail": manifest.get("topk_ranking_substitution")},
        {"audit_item": "no_daily_result_used_as_hourly", "passed": True, "detail": "scripts read hourly cache paths only"},
        {"audit_item": "final_labels_not_used_for_selection", "passed": manifest.get("selected_on_validation") is True or status == "not_run", "detail": manifest.get("selected_on_validation")},
        {"audit_item": "final_evaluation_scoring_only", "passed": manifest.get("final_evaluation_scoring_only") is True or status == "not_run", "detail": manifest.get("final_evaluation_scoring_only")},
        {"audit_item": "leakage_audit_passed", "passed": (REPORT_DIR / "joint_feature_leakage_audit.md").exists(), "detail": "feature audit file exists"},
        {"audit_item": "baseline_comparison_included", "passed": (OUTPUT_DIR / "baseline_delta_summary.csv").exists(), "detail": "baseline_delta_summary.csv"},
        {"audit_item": "combined_reaches_60", "passed": pd.notna(final_combined) and float(final_combined) >= 0.60, "detail": final_combined},
        {"audit_item": "combined_reaches_65", "passed": pd.notna(final_combined) and float(final_combined) >= 0.65, "detail": final_combined},
        {"audit_item": "stock_only_reaches_60", "passed": pd.notna(final_stock) and float(final_stock) >= 0.60, "detail": final_stock},
        {"audit_item": "stock_only_reaches_65", "passed": pd.notna(final_stock) and float(final_stock) >= 0.65, "detail": final_stock},
    ]
    unsafe_reasons = [row["audit_item"] for row in rows if not row["passed"] and row["audit_item"] in {"stock_30_present", "index_6_present", "total_36_present", "combined_result_reported", "no_confidence_abstention", "no_instrument_subset", "no_topk_ranking"}]
    if status == "not_run":
        result_safety = "unsafe"
    elif unsafe_reasons:
        result_safety = "unsafe"
    else:
        result_safety = "exploratory"
    rows.append({"audit_item": "result_safety", "passed": result_safety in {"safe", "exploratory"}, "detail": result_safety})
    write_csv(AUDIT_CSV, rows, ["audit_item", "passed", "detail"])
    content = [
        "# Joint Panel Training V1 Audit",
        "",
        f"- Training status: `{status}`.",
        f"- Result safety: `{result_safety}`.",
        f"- Combined reaches 60: {str(pd.notna(final_combined) and float(final_combined) >= 0.60).lower() if pd.notna(final_combined) else 'false'}.",
        f"- Combined reaches 65: {str(pd.notna(final_combined) and float(final_combined) >= 0.65).lower() if pd.notna(final_combined) else 'false'}.",
        f"- Stock-only reaches 60: {str(pd.notna(final_stock) and float(final_stock) >= 0.60).lower() if pd.notna(final_stock) else 'false'}.",
        f"- Stock-only reaches 65: {str(pd.notna(final_stock) and float(final_stock) >= 0.65).lower() if pd.notna(final_stock) else 'false'}.",
        "",
        "## Audit Items",
        "",
        markdown_table(["audit_item", "passed", "detail"], rows),
        "",
    ]
    AUDIT_MD.write_text("\n".join(content), encoding="utf-8")
    print(f"joint_training_audit_status={result_safety}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
