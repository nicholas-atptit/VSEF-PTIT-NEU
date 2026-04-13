"""Artifact naming, persistence helpers, and manifests for ML models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


ARTIFACT_SCHEMA_VERSION = 2
MODEL_EXTENSIONS = {
    "cart": ".joblib",
    "lstm": ".pt",
    "bilstm": ".pt",
    "sarimax": ".joblib",
    "ets": ".joblib",
    "xgboost": ".joblib",
    "lightgbm": ".joblib",
    "stacking": ".joblib",
}
TASK_PREFIXES = {
    "trend": "trend_classifier",
    "return": "return_regressor",
}


def model_extension(algorithm: str) -> str:
    algo = algorithm.lower()
    if algo not in MODEL_EXTENSIONS:
        raise ValueError(f"Unsupported algorithm '{algorithm}'")
    return MODEL_EXTENSIONS[algo]


def artifact_filename(task: str, algorithm: str, horizon: str) -> str:
    if task not in TASK_PREFIXES:
        raise ValueError(f"Unsupported task '{task}'")
    return f"{TASK_PREFIXES[task]}_{algorithm.lower()}_{horizon.lower()}{model_extension(algorithm)}"


def artifact_path(model_root: Path, ticker: str, task: str, algorithm: str, horizon: str) -> Path:
    return model_root / ticker.upper() / artifact_filename(task=task, algorithm=algorithm, horizon=horizon)


def meta_path(model_path: Path) -> Path:
    return model_path.with_suffix(".meta.joblib")


def scaler_path(model_path: Path) -> Path:
    return model_path.with_suffix(".scaler.joblib")


def manifest_path(model_root: Path, ticker: str) -> Path:
    return model_root / ticker.upper() / "manifest.json"


def ensure_ticker_dir(model_root: Path, ticker: str) -> Path:
    ticker_dir = model_root / ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)
    return ticker_dir


def write_manifest(model_root: Path, ticker: str, manifest: dict) -> Path:
    path = manifest_path(model_root, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return path


def load_manifest(model_root: Path, ticker: str) -> dict:
    path = manifest_path(model_root, ticker)
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest found for {ticker}. Expected {path}. Run training first."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cleanup_ticker_dir(model_root: Path, ticker: str) -> None:
    """Delete stale model artifacts for a ticker before writing a fresh bundle."""

    ticker_dir = ensure_ticker_dir(model_root, ticker)
    for pattern in ("*.joblib", "*.pt", "*.json", "*.csv"):
        for file_path in ticker_dir.glob(pattern):
            file_path.unlink()


def relative_paths(paths: Iterable[Path], root: Path) -> list[str]:
    return [str(path.relative_to(root)) for path in paths]
