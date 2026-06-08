"""Run Track B current broader-pipeline model comparison from current artifacts."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.vn30_hourly_dual_track_common import (
    CURRENT_RF_H60,
    EXPANDED_OUTPUT_DIR,
    HORIZONS,
    REPO_ROOT,
    RESULT_COLUMNS_CURRENT,
    STACKING_OUTPUT_DIR,
    active_stock_tickers,
    markdown_table,
    pct,
    rel,
    write_csv,
    write_json,
)

OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_current_pipeline_model_comparison"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_current_pipeline_model_comparison"


def convert_candidate(row: pd.Series) -> dict[str, object]:
    final_accuracy = float(row["final_accuracy"])
    return {
        "track": "current_pipeline",
        "model": row["model_name"],
        "horizon": int(float(row["horizon"])),
        "feature_set": row["feature_set"],
        "validation_accuracy": float(row["validation_accuracy"]),
        "final_accuracy": final_accuracy,
        "final_rows": int(float(row["final_rows"])),
        "final_coverage": float(row["final_coverage"]),
        "delta_vs_current_rf_h60_56_12": final_accuracy - CURRENT_RF_H60,
        "pass_current_baseline": final_accuracy > CURRENT_RF_H60,
        "pass_60": final_accuracy >= 0.60,
        "pass_65": final_accuracy >= 0.65,
        "selected_on_validation": False,
        "claim_level": "exploratory",
    }


def convert_ensemble(row: pd.Series) -> dict[str, object]:
    final_accuracy = float(row["final_accuracy"])
    return {
        "track": "current_pipeline",
        "model": row["ensemble_method"],
        "horizon": int(float(row["horizon"])),
        "feature_set": "validation_selected_ensemble",
        "validation_accuracy": float(row["validation_accuracy"]),
        "final_accuracy": final_accuracy,
        "final_rows": int(float(row["final_rows"])),
        "final_coverage": float(row["final_coverage"]),
        "delta_vs_current_rf_h60_56_12": final_accuracy - CURRENT_RF_H60,
        "pass_current_baseline": final_accuracy > CURRENT_RF_H60,
        "pass_60": final_accuracy >= 0.60,
        "pass_65": final_accuracy >= 0.65,
        "selected_on_validation": False,
        "claim_level": "exploratory",
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_path = EXPANDED_OUTPUT_DIR / "final_candidate_results.csv"
    ensemble_path = STACKING_OUTPUT_DIR / "final_ensemble_results.csv"
    if not candidate_path.exists():
        raise FileNotFoundError(f"Missing current expanded output: {rel(candidate_path)}")
    candidates = pd.read_csv(candidate_path)
    candidates = candidates[candidates["model_family"].ne("Baseline")].copy()
    rows = [convert_candidate(row) for _idx, row in candidates.iterrows()]
    ensemble_rows: list[dict[str, object]] = []
    if ensemble_path.exists():
        ensembles = pd.read_csv(ensemble_path)
        ensemble_rows = [convert_ensemble(row) for _idx, row in ensembles.iterrows()]
        rows.extend(ensemble_rows)
    valid = [row for row in rows if math.isfinite(float(row["validation_accuracy"]))]
    selected = max(valid, key=lambda row: (float(row["validation_accuracy"]), float(row["final_accuracy"]))) if valid else None
    if selected:
        selected["selected_on_validation"] = True
    reference = [
        row for row in rows
        if row["model"] == "random_forest"
        and int(row["horizon"]) == 60
        and row["feature_set"] == "stock_lagged_rolling_plus_index_context"
    ]
    reproduced = bool(reference and abs(float(reference[0]["final_accuracy"]) - CURRENT_RF_H60) <= 0.0005 and int(reference[0]["final_rows"]) == 8637)
    write_json(OUTPUT_DIR / "run_config.json", {"track": "current_pipeline", "source": rel(EXPANDED_OUTPUT_DIR), "ensemble_source": rel(STACKING_OUTPUT_DIR), "horizons": HORIZONS, "current_rf_h60_baseline": CURRENT_RF_H60, "data_fetch": False, "confidence_abstention": False})
    write_json(OUTPUT_DIR / "current_pipeline_comparison_manifest.json", {"track": "current_pipeline", "status": "completed", "active_ticker_count": len(active_stock_tickers()), "rf_h60_reproduced": reproduced, "current_rf_h60_baseline": CURRENT_RF_H60, "selected_on_validation": bool(selected)})
    write_csv(OUTPUT_DIR / "validation_candidate_results.csv", rows, RESULT_COLUMNS_CURRENT)
    write_csv(OUTPUT_DIR / "final_candidate_results.csv", rows, RESULT_COLUMNS_CURRENT)
    write_csv(OUTPUT_DIR / "selected_candidate_summary.csv", [selected] if selected else [], RESULT_COLUMNS_CURRENT)
    write_csv(OUTPUT_DIR / "current_rf_h60_reference_row.csv", reference, RESULT_COLUMNS_CURRENT)
    write_csv(OUTPUT_DIR / "model_comparison_summary.csv", rows, RESULT_COLUMNS_CURRENT)
    write_csv(OUTPUT_DIR / "ensemble_summary.csv", ensemble_rows, RESULT_COLUMNS_CURRENT)
    for name in ["run_config.json", "current_pipeline_comparison_manifest.json"]:
        source = OUTPUT_DIR / name
        target = REPORT_DIR / name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    for name in ["validation_candidate_results.csv", "final_candidate_results.csv", "selected_candidate_summary.csv", "current_rf_h60_reference_row.csv", "model_comparison_summary.csv", "ensemble_summary.csv"]:
        source = OUTPUT_DIR / name
        target = REPORT_DIR / name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    best_final = max(rows, key=lambda row: float(row["final_accuracy"])) if rows else {}
    report_rows = [
        {"metric": "rf_h60_reproduced", "value": str(reproduced).lower()},
        {"metric": "selected_model", "value": selected.get("model", "") if selected else ""},
        {"metric": "selected_final_accuracy", "value": pct(selected.get("final_accuracy", math.nan)) if selected else ""},
        {"metric": "best_final_model", "value": best_final.get("model", "")},
        {"metric": "best_final_accuracy", "value": pct(best_final.get("final_accuracy", math.nan))},
    ]
    (REPORT_DIR / "current_pipeline_comparison_report.md").write_text("# Track B Current Pipeline Model Comparison\n\n" + markdown_table(["metric", "value"], report_rows) + "\n", encoding="utf-8")
    print(f"current_pipeline_comparison_status=completed selected={selected.get('model') if selected else ''} output_dir={rel(OUTPUT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
