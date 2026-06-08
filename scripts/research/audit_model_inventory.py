"""Audit repository references to model families used in VN30 screening."""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCAN_DIRS = ["src", "scripts", "tests", "configs", "config", "reports"]
OUT_DIR = REPO_ROOT / "reports" / "generated" / "model_inventory"
CSV_PATH = OUT_DIR / "model_inventory.csv"
MD_PATH = OUT_DIR / "model_inventory.md"

MODEL_TERMS = {
    "Random Forest": [r"RandomForest", r"random_forest"],
    "ExtraTrees": [r"ExtraTrees", r"extra_trees", r"extratrees"],
    "Decision Tree / CART": [r"DecisionTree", r"\bCART\b", r"decision_tree"],
    "XGBoost": [r"XGBoost", r"\bXGB", r"xgboost"],
    "LightGBM": [r"LightGBM", r"\bLGBM", r"lightgbm"],
    "CatBoost": [r"CatBoost", r"catboost"],
    "Logistic Regression": [r"LogisticRegression", r"logistic_regression"],
    "SVM / SVC": [r"\bSVM\b", r"\bSVC\b", r"support_vector"],
    "SARIMAX": [r"SARIMAX", r"sarimax"],
    "ARIMA": [r"\bARIMA\b", r"arima"],
    "ETS": [r"\bETS\b", r"ExponentialSmoothing", r"exponential_smoothing"],
    "LSTM": [r"\bLSTM\b", r"lstm"],
    "BiLSTM": [r"BiLSTM", r"bilstm"],
    "GRU": [r"\bGRU\b", r"gru"],
    "Transformer": [r"Transformer", r"transformer"],
    "Stacking": [r"Stacking", r"stacking"],
    "Voting": [r"Voting", r"voting"],
    "Ensemble": [r"Ensemble", r"ensemble"],
    "baseline": [r"baseline"],
    "moving_average": [r"moving_average", r"moving average"],
    "previous_direction": [r"previous_direction", r"previous direction"],
    "always_up": [r"always_up", r"always-up", r"always up"],
    "random_direction": [r"random_direction", r"random_seeded_direction"],
}

ACTIVE_READY = {"Random Forest", "ExtraTrees", "Decision Tree / CART", "XGBoost", "LightGBM", "Logistic Regression", "Stacking", "Voting", "Ensemble", "baseline", "moving_average", "previous_direction", "always_up", "random_direction"}
USABLE_NEEDS_INTEGRATION = {"SARIMAX", "ARIMA", "ETS", "SVM / SVC", "CatBoost"}
LEGACY_OR_HEAVY = {"LSTM", "BiLSTM", "GRU", "Transformer"}


@dataclass
class Hit:
    model_family: str
    path: str
    line: int
    snippet: str


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def iter_text_files() -> list[Path]:
    files: list[Path] = []
    for directory in SCAN_DIRS:
        root = REPO_ROOT / directory
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".md", ".json", ".yaml", ".yml", ".csv", ".txt"}:
                files.append(path)
    return files


def scan_hits() -> list[Hit]:
    compiled = {
        family: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        for family, patterns in MODEL_TERMS.items()
    }
    hits: list[Hit] = []
    for path in iter_text_files():
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, start=1):
            for family, patterns in compiled.items():
                if any(pattern.search(line) for pattern in patterns):
                    hits.append(Hit(family, rel(path), line_no, line.strip()[:240]))
    return hits


def classify(family: str, hits: list[Hit]) -> dict[str, Any]:
    count = len([hit for hit in hits if hit.model_family == family])
    paths = sorted({hit.path for hit in hits if hit.model_family == family})
    if count == 0:
        status = "broken_or_missing"
    elif family in ACTIVE_READY:
        status = "active_ready"
    elif family in USABLE_NEEDS_INTEGRATION:
        status = "usable_needs_integration"
    elif family in LEGACY_OR_HEAVY:
        status = "legacy_only"
    else:
        status = "usable_needs_integration"
    screening = status in {"active_ready", "usable_needs_integration"} and family not in {"Stacking", "Voting", "Ensemble"}
    stacking = family in {"Random Forest", "ExtraTrees", "Decision Tree / CART", "XGBoost", "LightGBM", "Logistic Regression"} and status == "active_ready"
    if family in {"SARIMAX", "ARIMA", "ETS", "CatBoost", "SVM / SVC"}:
        stacking = False
    return {
        "model_family": family,
        "reference_count": count,
        "example_paths": "; ".join(paths[:8]),
        "status": status,
        "candidate_for_screening": "yes" if screening else "no",
        "candidate_for_stacking": "yes" if stacking else "no",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model_family", "reference_count", "example_paths", "status", "candidate_for_screening", "candidate_for_stacking"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    fields = ["model_family", "reference_count", "status", "candidate_for_screening", "candidate_for_stacking"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")).replace("|", "\\|") for field in fields) + " |")
    return "\n".join(lines)


def main() -> int:
    hits = scan_hits()
    rows = [classify(family, hits) for family in MODEL_TERMS]
    write_csv(CSV_PATH, rows)
    content = [
        "# Model Inventory Audit",
        "",
        f"- Scanned directories: {', '.join(SCAN_DIRS)}.",
        f"- Total matched references: {len(hits)}.",
        "- Classification is repository-inventory evidence, not model performance evidence.",
        "",
        markdown_table(rows),
        "",
    ]
    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text("\n".join(content), encoding="utf-8")
    print(f"model_inventory_csv={rel(CSV_PATH)}")
    print(f"model_inventory_md={rel(MD_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
