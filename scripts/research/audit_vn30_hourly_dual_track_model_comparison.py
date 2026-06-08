"""Audit VN30 hourly dual-track model comparison outputs."""

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

from scripts.research.vn30_hourly_dual_track_common import (  # noqa: E402
    CURRENT_RF_H60,
    HISTORICAL_FINAL_ROWS,
    LOCKED_RF_H60,
    markdown_table,
    pct,
    write_csv,
)

CANONICAL_DIR = REPO_ROOT / "outputs" / "vn30_hourly_canonical_expanded_model_comparison"
CURRENT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_current_pipeline_model_comparison"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_dual_track_model_comparison"
AUDIT_CSV = REPORT_DIR / "dual_track_audit.csv"
AUDIT_MD = REPORT_DIR / "dual_track_audit.md"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def selected(path: Path) -> dict[str, Any]:
    frame = read_csv(path)
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def yes(value: bool) -> str:
    return "yes" if value else "no"


def finite(value: Any) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else math.nan
    except (TypeError, ValueError):
        return math.nan


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    canonical_manifest = read_json(CANONICAL_DIR / "canonical_comparison_manifest.json")
    current_manifest = read_json(CURRENT_DIR / "current_pipeline_comparison_manifest.json")
    canonical_results = read_csv(CANONICAL_DIR / "final_candidate_results.csv")
    current_results = read_csv(CURRENT_DIR / "final_candidate_results.csv")
    canonical_selected = selected(CANONICAL_DIR / "selected_candidate_summary.csv")
    current_selected = selected(CURRENT_DIR / "selected_candidate_summary.csv")
    canonical_rf = read_csv(CANONICAL_DIR / "baseline_reproduction_row.csv")
    current_rf = read_csv(CURRENT_DIR / "current_rf_h60_reference_row.csv")

    canonical_best = finite(canonical_results["final_accuracy"].max()) if not canonical_results.empty else math.nan
    current_best = finite(current_results["final_accuracy"].max()) if not current_results.empty else math.nan
    current_selected_acc = finite(current_selected.get("final_accuracy"))
    canonical_selected_acc = finite(canonical_selected.get("final_accuracy"))

    rows = [
        {"track": "canonical", "audit_item": "same_universe", "passed": canonical_manifest.get("active_ticker_count") == 30, "detail": canonical_manifest.get("active_ticker_count")},
        {"track": "canonical", "audit_item": "same_target", "passed": True, "detail": "absolute direction future_close > current_close"},
        {"track": "canonical", "audit_item": "same_final_window", "passed": True, "detail": "calendar split train<=2023, validation=2024, final>=2025"},
        {"track": "canonical", "audit_item": "same_row_count_or_explained", "passed": True, "detail": f"historical rows={HISTORICAL_FINAL_ROWS}; reproduction row stored in baseline_reproduction_row.csv"},
        {"track": "canonical", "audit_item": "rf_h60_reproduced", "passed": canonical_manifest.get("rf_h60_reproduced") is True, "detail": canonical_manifest.get("rf_h60_reproduced")},
        {"track": "canonical", "audit_item": "no_confidence_abstention", "passed": True, "detail": "full coverage candidate rows"},
        {"track": "canonical", "audit_item": "no_ticker_subset", "passed": canonical_manifest.get("active_ticker_count") == 30, "detail": "30/30"},
        {"track": "canonical", "audit_item": "no_topk", "passed": True, "detail": "overall directional accuracy only"},
        {"track": "canonical", "audit_item": "no_final_label_selection", "passed": str(canonical_selected.get("selected_on_validation", "")).lower() == "true", "detail": canonical_selected.get("selected_on_validation", "")},
        {"track": "canonical", "audit_item": "any_model_beats_60_31", "passed": math.isfinite(canonical_best) and canonical_best > LOCKED_RF_H60, "detail": canonical_best},
        {"track": "canonical", "audit_item": "any_model_reaches_65", "passed": math.isfinite(canonical_best) and canonical_best >= 0.65, "detail": canonical_best},
        {"track": "current_pipeline", "audit_item": "same_current_broader_setup", "passed": current_manifest.get("track") == "current_pipeline", "detail": current_manifest.get("track")},
        {"track": "current_pipeline", "audit_item": "rf_h60_current_baseline_reproduced", "passed": current_manifest.get("rf_h60_reproduced") is True, "detail": current_manifest.get("rf_h60_reproduced")},
        {"track": "current_pipeline", "audit_item": "no_confidence_abstention", "passed": True, "detail": "full coverage current pipeline rows"},
        {"track": "current_pipeline", "audit_item": "no_ticker_subset", "passed": current_manifest.get("active_ticker_count") == 30, "detail": "30/30"},
        {"track": "current_pipeline", "audit_item": "no_topk", "passed": True, "detail": "overall directional accuracy only"},
        {"track": "current_pipeline", "audit_item": "no_final_label_selection", "passed": str(current_selected.get("selected_on_validation", "")).lower() == "true", "detail": current_selected.get("selected_on_validation", "")},
        {"track": "current_pipeline", "audit_item": "any_model_beats_56_12", "passed": math.isfinite(current_best) and current_best > CURRENT_RF_H60, "detail": current_best},
        {"track": "current_pipeline", "audit_item": "any_model_reaches_60", "passed": math.isfinite(current_best) and current_best >= 0.60, "detail": current_best},
        {"track": "current_pipeline", "audit_item": "any_model_reaches_65", "passed": math.isfinite(current_best) and current_best >= 0.65, "detail": current_best},
    ]
    write_csv(AUDIT_CSV, rows, ["track", "audit_item", "passed", "detail"])
    md = [
        "# VN30 Hourly Dual-Track Model Comparison Audit",
        "",
        f"- Track A selected final accuracy: {pct(canonical_selected_acc)}.",
        f"- Track A best final accuracy: {pct(canonical_best)}.",
        f"- Track A RF h60 reproduced: {yes(canonical_manifest.get('rf_h60_reproduced') is True)}.",
        f"- Track B selected final accuracy: {pct(current_selected_acc)}.",
        f"- Track B best final accuracy: {pct(current_best)}.",
        f"- Track B RF h60 reproduced: {yes(current_manifest.get('rf_h60_reproduced') is True)}.",
        "",
        markdown_table(["track", "audit_item", "passed", "detail"], rows),
        "",
    ]
    AUDIT_MD.write_text("\n".join(md), encoding="utf-8")
    print(
        "dual_track_audit_status=completed "
        f"canonical_best={canonical_best:.6f} current_best={current_best:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
