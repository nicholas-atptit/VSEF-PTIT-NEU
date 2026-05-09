"""Config-driven experiment orchestration for VSEF Phase 1."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import traceback
import uuid
from datetime import UTC, date, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in incomplete environments
    yaml = None

from src.data.adapters.vnstock_adapter import VnstockAdapter
from src.ml.baselines import BaselineRegistry
from src.ml.evaluation import MetricsEngine
from src.ml.evaluation.metrics_engine import METRICS_COLUMNS
from src.ml.models.factory import create_model


REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_V1_MODELS = {"sarimax", "ets", "xgboost", "lightgbm", "lstm", "bilstm", "stacking"}
REQUIRED_OHLCV_SCHEMA = {"date", "ticker", "open", "high", "low", "close", "volume"}
PREDICTION_COLUMNS = [
    "date",
    "ticker",
    "horizon",
    "model_name",
    "model_type",
    "y_true",
    "y_pred",
    "predicted_direction",
    "actual_direction",
    "notes",
]
EXPECTED_ARTIFACTS = [
    "forecast_summary.csv",
    "model_consensus_summary.csv",
    "model_health_summary.csv",
    "risk_summary.csv",
    "strategy_metrics.csv",
    "decision_lane_candidates.csv",
    "analysis_packets.jsonl",
]


class ExperimentOrchestrator:
    """Run a standardized experiment from a single YAML configuration."""

    def __init__(self, config_path: str) -> None:
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = (Path.cwd() / self.config_path).resolve()
        self.config: dict[str, Any] = {}
        self.output_dir: Path | None = None
        self.run_id = ""
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.predictions: pd.DataFrame = pd.DataFrame(columns=PREDICTION_COLUMNS)
        self.model_predictions: pd.DataFrame = pd.DataFrame(columns=PREDICTION_COLUMNS)
        self.baseline_predictions: pd.DataFrame = pd.DataFrame(columns=PREDICTION_COLUMNS)
        self.metrics: pd.DataFrame = pd.DataFrame(columns=METRICS_COLUMNS)
        self.provider_environment: dict[str, Any] = {}
        self.baseline_registry = BaselineRegistry()
        self.metrics_engine = MetricsEngine()

    def load_config(self) -> dict[str, Any]:
        """Load the YAML experiment configuration."""
        if yaml is None:
            raise RuntimeError("PyYAML is required to run YAML experiment configs. Install pyyaml in this environment.")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Experiment config must be a YAML mapping")
        self.config = loaded
        return self.config

    def validate_config(self) -> None:
        """Validate required config fields and Phase 0 governance boundaries."""
        if not self.config:
            self.load_config()

        required_paths = [
            ("experiment", "id"),
            ("data", "provider"),
            ("data", "frequency"),
            ("data", "universe"),
            ("data", "start_date"),
            ("data", "end_date"),
            ("target", "horizons"),
            ("models", "include"),
            ("baselines", "include"),
            ("evaluation", "method"),
            ("evaluation", "metrics"),
            ("outputs", "root_dir"),
        ]
        missing = [self._path_label(path) for path in required_paths if self._get_path(path) in (None, "")]
        if missing:
            raise ValueError(f"Experiment config is missing required fields: {missing}")

        provider = str(self._get_path(("data", "provider"))).strip()
        if provider != "vnstock_data":
            raise ValueError("data.provider must be vnstock_data for VSEF v1")

        frequency = str(self._get_path(("data", "frequency"))).strip().lower()
        if frequency != "daily":
            raise ValueError("data.frequency must be daily for VSEF v1")

        universe = self._get_path(("data", "universe"))
        if not isinstance(universe, list) or not universe or not all(str(ticker).strip() for ticker in universe):
            raise ValueError("data.universe must be a non-empty list of ticker symbols")

        for field in ("start_date", "end_date"):
            self._parse_iso_date(str(self._get_path(("data", field))), f"data.{field}")
        for field in ("train_start", "train_end", "test_start", "test_end"):
            value = self._get_path(("evaluation", field))
            if value is not None:
                self._parse_iso_date(str(value), f"evaluation.{field}")

        horizons = self._get_path(("target", "horizons"))
        if not isinstance(horizons, list) or not horizons:
            raise ValueError("target.horizons must be a non-empty list")
        if any((not isinstance(horizon, int)) or horizon <= 0 for horizon in horizons):
            raise ValueError("target.horizons must contain positive integers")

        models = [str(name).strip().lower() for name in self._get_path(("models", "include"))]
        unsupported_models = sorted(set(models) - FROZEN_V1_MODELS)
        if unsupported_models:
            raise ValueError(
                "models.include contains models outside the Phase 0 frozen registry: "
                f"{unsupported_models}. Supported: {sorted(FROZEN_V1_MODELS)}"
            )

        baselines = [str(name).strip().lower() for name in self._get_path(("baselines", "include"))]
        unsupported_baselines = sorted(set(baselines) - set(self.baseline_registry.list_baselines()))
        if unsupported_baselines:
            raise ValueError(
                "baselines.include contains unsupported baselines: "
                f"{unsupported_baselines}. Supported: {self.baseline_registry.list_baselines()}"
            )

        schema = self._get_path(("data", "schema")) or []
        if schema:
            missing_schema = REQUIRED_OHLCV_SCHEMA - {str(column).strip() for column in schema}
            if missing_schema:
                raise ValueError(f"data.schema is missing required OHLCV fields: {sorted(missing_schema)}")

        root_dir = self._resolve_repo_path(str(self._get_path(("outputs", "root_dir"))))
        expected_root = (REPO_ROOT / "outputs" / "experiments").resolve()
        allow_override = bool(self._get_path(("runtime", "allow_test_output_override")) or False)
        if not allow_override and not self._is_relative_to(root_dir, expected_root):
            raise ValueError("outputs.root_dir must be under outputs/experiments unless explicitly overridden for testing")

        max_workers = int(self._get_path(("runtime", "max_workers")) or 1)
        if max_workers != 1:
            self.warnings.append("runtime.max_workers is accepted but ExperimentOrchestrator v1 executes sequentially")

    def prepare_output_dir(self) -> Path:
        """Create the standardized experiment output folder."""
        if not self.config:
            self.load_config()
        experiment_id = str(self._get_path(("experiment", "id"))).strip()
        root_dir = self._resolve_repo_path(str(self._get_path(("outputs", "root_dir"))))
        self.output_dir = root_dir / experiment_id
        for relative in (
            "config",
            "manifests",
            "logs",
            "metrics",
            "predictions",
            "artifacts",
            "charts",
            "reports",
        ):
            (self.output_dir / relative).mkdir(parents=True, exist_ok=True)
        (self.output_dir / "charts" / ".gitkeep").write_text("", encoding="utf-8")
        (self.output_dir / "artifacts" / ".gitkeep").write_text("", encoding="utf-8")
        return self.output_dir

    def save_resolved_config(self) -> None:
        """Save original and resolved config copies into the run output folder."""
        if self.output_dir is None:
            self.prepare_output_dir()
        assert self.output_dir is not None
        shutil.copyfile(self.config_path, self.output_dir / "config" / "original_config.yaml")
        with (self.output_dir / "config" / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.config, handle, sort_keys=False)

    def run(self) -> dict[str, Any]:
        """Execute the configured experiment and write standard artifacts."""
        status = "running"

        try:
            self.load_config()
            self.started_at = datetime.now(UTC)
            self.run_id = self._new_run_id()
            self.validate_config()
            self.prepare_output_dir()
            self.save_resolved_config()
            self._initialize_logs()
            self._log("run.log", f"Experiment {self._experiment_id()} started with run_id={self.run_id}")
            self.write_manifest(status="running")

            fail_fast = bool(self._get_path(("runtime", "fail_fast")) or False)
            results: list[dict[str, Any]] = []
            for ticker in self.config["data"]["universe"]:
                for horizon in self.config["target"]["horizons"]:
                    try:
                        results.append(self.run_ticker_horizon(str(ticker).upper().strip(), int(horizon)))
                    except Exception as exc:
                        self._record_error(
                            stage="ticker_horizon",
                            message=str(exc),
                            ticker=str(ticker).upper().strip(),
                            horizon=int(horizon),
                            traceback_text=traceback.format_exc(),
                        )
                        if fail_fast:
                            raise

            self._finalize_predictions()
            self._write_prediction_outputs()
            self.metrics = self.compute_metrics(self.predictions)
            self._write_metrics_outputs()
            self.write_summary()

            if self.errors and self.predictions.empty:
                status = "failed"
            elif self.errors:
                status = "completed_with_errors"
            else:
                status = "completed"
            self.completed_at = datetime.now(UTC)
            self.write_manifest(status=status, errors=self.errors)
            self._log("run.log", f"Experiment {self._experiment_id()} completed with status={status}")
            return {
                "status": status,
                "run_id": self.run_id,
                "output_dir": str(self.output_dir),
                "results": results,
                "errors": self.errors,
                "warnings": self.warnings,
            }
        except Exception as exc:
            if self.output_dir is None:
                try:
                    self.prepare_output_dir()
                except Exception:
                    pass
            self._record_error(
                stage="run",
                message=str(exc),
                traceback_text=traceback.format_exc(),
            )
            self.completed_at = datetime.now(UTC)
            self._write_empty_outputs_if_needed()
            self.write_summary()
            self.write_manifest(status="failed", errors=self.errors)
            return {
                "status": "failed",
                "run_id": self.run_id,
                "output_dir": str(self.output_dir) if self.output_dir else "",
                "errors": self.errors,
                "warnings": self.warnings,
            }

    def run_ticker_horizon(self, ticker: str, horizon: int) -> dict[str, Any]:
        """Run all configured models and baselines for one ticker/horizon pair."""
        data = self._fetch_ohlcv(ticker)
        self._validate_ohlcv_frame(data, ticker)

        model_frames = self.run_models(ticker, horizon, data=data)
        baseline_frames = self.run_baselines(ticker, horizon, data=data)

        model_frame = self._concat_frames(model_frames, PREDICTION_COLUMNS)
        baseline_frame = self._concat_frames(baseline_frames, PREDICTION_COLUMNS)
        self.model_predictions = self._append_frame(self.model_predictions, model_frame)
        self.baseline_predictions = self._append_frame(self.baseline_predictions, baseline_frame)
        return {
            "ticker": ticker,
            "horizon": horizon,
            "model_prediction_rows": int(len(model_frame)),
            "baseline_prediction_rows": int(len(baseline_frame)),
        }

    def run_models(self, ticker: str, horizon: int, data: pd.DataFrame | None = None) -> list[pd.DataFrame]:
        """Run configured Phase 0-supported models for a ticker/horizon pair."""
        if not bool(self._get_path(("models", "enabled"))):
            self.warnings.append("models.enabled is false; model execution skipped")
            return []
        if data is None:
            data = self._fetch_ohlcv(ticker)
        supervised = self._build_supervised_frame(data, horizon)
        train_frame, test_frame = self._split_supervised(supervised)

        frames: list[pd.DataFrame] = []
        fail_fast = bool(self._get_path(("runtime", "fail_fast")) or False)
        for model_name in [str(name).strip().lower() for name in self.config["models"]["include"]]:
            try:
                frames.append(self._run_single_model(model_name, train_frame, test_frame, ticker, horizon))
            except Exception as exc:
                self._record_error(
                    stage="model",
                    message=str(exc),
                    ticker=ticker,
                    horizon=horizon,
                    model_name=model_name,
                    traceback_text=traceback.format_exc(),
                )
                if fail_fast:
                    raise
        return frames

    def run_baselines(self, ticker: str, horizon: int, data: pd.DataFrame | None = None) -> list[pd.DataFrame]:
        """Run configured deterministic baselines for a ticker/horizon pair."""
        if not bool(self._get_path(("baselines", "enabled"))):
            self.warnings.append("baselines.enabled is false; baseline execution skipped")
            return []
        if data is None:
            data = self._fetch_ohlcv(ticker)
        frames: list[pd.DataFrame] = []
        fail_fast = bool(self._get_path(("runtime", "fail_fast")) or False)
        baseline_config = {
            **self.config,
            "seed": int(self._get_path(("experiment", "seed")) or 42),
        }
        test_start = self._parse_iso_date(str(self._get_path(("evaluation", "test_start"))), "evaluation.test_start")
        test_end = self._parse_iso_date(str(self._get_path(("evaluation", "test_end"))), "evaluation.test_end")
        for baseline_name in [str(name).strip().lower() for name in self.config["baselines"]["include"]]:
            try:
                frame = self.baseline_registry.run_baseline(baseline_name, data, horizon, baseline_config)
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
                frame = frame[(frame["date"].dt.date >= test_start) & (frame["date"].dt.date <= test_end)]
                frames.append(frame[PREDICTION_COLUMNS].reset_index(drop=True))
            except Exception as exc:
                self._record_error(
                    stage="baseline",
                    message=str(exc),
                    ticker=ticker,
                    horizon=horizon,
                    model_name=baseline_name,
                    traceback_text=traceback.format_exc(),
                )
                if fail_fast:
                    raise
        return frames

    def compute_metrics(self, predictions: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
        """Compute standardized metrics from prediction rows."""
        frame = pd.DataFrame(predictions) if isinstance(predictions, list) else predictions
        return self.metrics_engine.compute(
            frame,
            experiment_id=self._experiment_id(),
            run_id=self.run_id,
            start_date=str(self._get_path(("evaluation", "test_start")) or ""),
            end_date=str(self._get_path(("evaluation", "test_end")) or ""),
        )

    def write_manifest(self, status: str, errors: list | None = None) -> None:
        """Write the run manifest."""
        if self.output_dir is None:
            return
        manifest = self._manifest_payload(status=status, errors=list(errors or []))
        self._write_json(self.output_dir / "manifests" / "run_manifest.json", manifest)

    def write_summary(self) -> None:
        """Write a markdown experiment summary."""
        if self.output_dir is None:
            return
        missing_artifacts = self._missing_expected_artifacts()
        lines = [
            f"# {self._experiment_id()} Summary",
            "",
            f"- Status: `{self._status_from_errors()}`",
            f"- Run ID: `{self.run_id}`",
            f"- Experiment name: {self._get_path(('experiment', 'name')) or ''}",
            f"- Phase: {self._get_path(('experiment', 'phase')) or ''}",
            f"- Provider: `{self._get_path(('data', 'provider'))}`",
            f"- Frequency: `{self._get_path(('data', 'frequency'))}`",
            f"- Universe: {', '.join(str(ticker) for ticker in self._get_path(('data', 'universe')) or [])}",
            f"- Horizons: {', '.join(str(h) for h in self._get_path(('target', 'horizons')) or [])}",
            f"- Models: {', '.join(str(model) for model in self._get_path(('models', 'include')) or [])}",
            f"- Baselines: {', '.join(str(model) for model in self._get_path(('baselines', 'include')) or [])}",
            "",
            "## Outputs",
            "",
            f"- Prediction rows: {len(self.predictions)}",
            f"- Metric rows: {len(self.metrics)}",
            f"- Error count: {len(self.errors)}",
            f"- Warning count: {len(self.warnings) + len(missing_artifacts)}",
            "",
        ]
        if self.errors:
            lines.extend(["## Errors", ""])
            for error in self.errors:
                context = ", ".join(
                    f"{key}={value}"
                    for key, value in error.items()
                    if key not in {"traceback", "provider_environment"} and value is not None
                )
                lines.append(f"- {context}")
            lines.append("")
        if self.provider_environment:
            lines.extend(
                [
                    "## Provider Environment",
                    "",
                    f"- Provider: `{self.provider_environment.get('provider_name')}`",
                    f"- Import status: `{self.provider_environment.get('import_status')}`",
                    f"- Python executable: `{self.provider_environment.get('python_executable')}`",
                    f"- Python version: `{self.provider_environment.get('python_version')}`",
                    f"- Import error message: `{self.provider_environment.get('import_error_message') or ''}`",
                    "",
                ]
            )
        if missing_artifacts:
            lines.extend(
                [
                    "## Missing Expected Artifacts",
                    "",
                    "These artifacts were not generated by the current runtime path and were not faked:",
                    "",
                ]
            )
            lines.extend(f"- `{name}`" for name in missing_artifacts)
            lines.append("")
        lines.extend(
            [
                "## Evidence",
                "",
                "- `config/original_config.yaml`",
                "- `config/resolved_config.yaml`",
                "- `manifests/run_manifest.json`",
                "- `logs/run.log`",
                "- `logs/errors.log`",
                "- `metrics/metrics.csv`",
                "- `metrics/metrics_summary.json`",
                "- `predictions/predictions.csv`",
            ]
        )
        (self.output_dir / "reports" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _run_single_model(
        self,
        model_name: str,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        ticker: str,
        horizon: int,
    ) -> pd.DataFrame:
        feature_columns = list(train_frame.attrs.get("feature_columns") or [])
        if not feature_columns:
            raise ValueError("No feature columns available for model execution")
        X_train = train_frame[feature_columns].to_numpy(dtype=float)
        y_train = train_frame["y_true"].to_numpy(dtype=float)
        X_test = test_frame[feature_columns].to_numpy(dtype=float)

        params = self._model_params(model_name)
        task = str(self._get_path(("target", "task_type")) or "regression")
        params.setdefault("task", task)
        params.setdefault("random_state", int(self._get_path(("experiment", "seed")) or 42))
        if model_name == "stacking":
            params.setdefault("base_learners", ["xgboost", "lightgbm", "sarimax", "ets"])
            params.setdefault("n_estimators", 20)
        if model_name in {"lstm", "bilstm"}:
            sequence_length = int(params.get("sequence_length", 5))
            X_train, y_train = self._sequence_xy(train_frame, feature_columns, sequence_length)
            X_test, test_frame = self._sequence_test_frame(train_frame, test_frame, feature_columns, sequence_length)
        model = create_model(model_name, **params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return self._prediction_frame(
            test_frame,
            ticker=ticker,
            horizon=horizon,
            model_name=model_name,
            model_type="model",
            y_pred=np.asarray(y_pred, dtype=float).reshape(-1),
            notes="model_run_completed",
        )

    def _fetch_ohlcv(self, ticker: str) -> pd.DataFrame:
        provider_name = str(self._get_path(("data", "provider")) or "vnstock_data")
        provider_environment = self._check_provider_import(provider_name)
        if provider_environment["import_status"] != "available":
            reason = "vnstock_data_not_installed" if provider_name == "vnstock_data" else "provider_import_failed"
            raise ValueError(f"No OHLCV data available for {ticker}: {reason}")
        adapter = VnstockAdapter(symbol_list=[ticker])
        frame = adapter.get_ohlcv(
            ticker,
            str(self._get_path(("data", "start_date"))),
            str(self._get_path(("data", "end_date"))),
            "1D",
        )
        if frame.empty:
            reason = frame.attrs.get("source_notes") or frame.attrs.get("source_availability") or "empty_ohlcv"
            raise ValueError(f"No OHLCV data available for {ticker}: {reason}")
        return frame

    def _build_supervised_frame(self, data: pd.DataFrame, horizon: int) -> pd.DataFrame:
        frame = data.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
        frame["future_close"] = frame["close"].shift(-int(horizon))
        target_column = str(self._get_path(("target", "column")) or "close").lower()
        task_type = str(self._get_path(("target", "task_type")) or "regression").lower()
        if target_column in {"forward_return", "future_return", "return"} or "return" in target_column:
            frame["y_true"] = (frame["future_close"] / frame["close"]) - 1.0
            frame["prediction_reference"] = 0.0
            frame["target_kind"] = "return"
        elif task_type == "classification" or "direction" in target_column:
            frame["y_true"] = np.where(frame["future_close"] > frame["close"], 1.0, 0.0)
            frame["prediction_reference"] = 0.5
            frame["target_kind"] = "direction"
        else:
            frame["y_true"] = frame["future_close"]
            frame["prediction_reference"] = frame["close"]
            frame["target_kind"] = "price"

        feature_columns = self._add_features(frame)
        frame = frame.dropna(subset=["y_true", *feature_columns]).reset_index(drop=True)
        frame.attrs["feature_columns"] = feature_columns
        if frame.empty:
            raise ValueError("No supervised rows available after applying horizon and feature filters")
        return frame

    def _add_features(self, frame: pd.DataFrame) -> list[str]:
        features_cfg = dict(self.config.get("features") or {})
        if not bool(features_cfg.get("enabled", True)):
            frame["feature_close"] = frame["close"]
            return ["feature_close"]

        feature_sets = [str(name) for name in features_cfg.get("feature_sets") or ["ohlcv_basic"]]
        columns: list[str] = []
        if "ohlcv_basic" in feature_sets:
            columns.extend(["open", "high", "low", "close", "volume"])
        if "returns_basic" in feature_sets:
            frame["return_1"] = frame["close"].pct_change().fillna(0.0)
            frame["high_low_range"] = (frame["high"] - frame["low"]) / frame["close"].replace(0.0, np.nan)
            frame["close_open_return"] = (frame["close"] - frame["open"]) / frame["open"].replace(0.0, np.nan)
            columns.extend(["return_1", "high_low_range", "close_open_return"])
        if "moving_average_basic" in feature_sets:
            frame["ma_3"] = frame["close"].rolling(window=3, min_periods=1).mean()
            frame["ma_5"] = frame["close"].rolling(window=5, min_periods=1).mean()
            columns.extend(["ma_3", "ma_5"])
        if not columns:
            frame["feature_close"] = frame["close"]
            columns.append("feature_close")
        return columns

    def _split_supervised(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_start = self._parse_iso_date(str(self._get_path(("evaluation", "train_start"))), "evaluation.train_start")
        train_end = self._parse_iso_date(str(self._get_path(("evaluation", "train_end"))), "evaluation.train_end")
        test_start = self._parse_iso_date(str(self._get_path(("evaluation", "test_start"))), "evaluation.test_start")
        test_end = self._parse_iso_date(str(self._get_path(("evaluation", "test_end"))), "evaluation.test_end")
        dates = frame["date"].dt.date
        train_frame = frame[(dates >= train_start) & (dates <= train_end)].copy()
        test_frame = frame[(dates >= test_start) & (dates <= test_end)].copy()
        train_frame.attrs["feature_columns"] = frame.attrs.get("feature_columns")
        test_frame.attrs["feature_columns"] = frame.attrs.get("feature_columns")
        if train_frame.empty:
            raise ValueError("No training rows available for configured train window")
        if test_frame.empty:
            raise ValueError("No test rows available for configured test window")
        return train_frame.reset_index(drop=True), test_frame.reset_index(drop=True)

    @staticmethod
    def _sequence_xy(
        frame: pd.DataFrame,
        feature_columns: list[str],
        sequence_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(frame) < sequence_length:
            raise ValueError("Not enough rows for configured sequence_length")
        X = frame[feature_columns].to_numpy(dtype=float)
        y = frame["y_true"].to_numpy(dtype=float)
        sequences = []
        targets = []
        for idx in range(sequence_length - 1, len(frame)):
            sequences.append(X[idx - sequence_length + 1 : idx + 1])
            targets.append(y[idx])
        return np.asarray(sequences, dtype=float), np.asarray(targets, dtype=float)

    @staticmethod
    def _sequence_test_frame(
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        feature_columns: list[str],
        sequence_length: int,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        context = pd.concat([train_frame.tail(sequence_length - 1), test_frame], ignore_index=True)
        X = context[feature_columns].to_numpy(dtype=float)
        sequences = []
        output_rows = []
        for idx in range(sequence_length - 1, len(context)):
            source_row = context.iloc[idx]
            if source_row["date"] in set(test_frame["date"]):
                sequences.append(X[idx - sequence_length + 1 : idx + 1])
                output_rows.append(source_row)
        if not sequences:
            raise ValueError("No test sequences available for configured sequence_length")
        return np.asarray(sequences, dtype=float), pd.DataFrame(output_rows).reset_index(drop=True)

    def _prediction_frame(
        self,
        test_frame: pd.DataFrame,
        *,
        ticker: str,
        horizon: int,
        model_name: str,
        model_type: str,
        y_pred: np.ndarray,
        notes: str,
    ) -> pd.DataFrame:
        if len(y_pred) != len(test_frame):
            raise ValueError(f"Prediction length mismatch for {model_name}: {len(y_pred)} != {len(test_frame)}")
        output = test_frame[["date", "ticker", "y_true", "prediction_reference", "target_kind"]].copy()
        output["ticker"] = ticker
        output["horizon"] = int(horizon)
        output["model_name"] = model_name
        output["model_type"] = model_type
        output["y_pred"] = y_pred
        if (output["target_kind"] == "return").all():
            output["actual_direction"] = np.sign(pd.to_numeric(output["y_true"], errors="coerce"))
            output["predicted_direction"] = np.sign(pd.to_numeric(output["y_pred"], errors="coerce"))
        elif (output["target_kind"] == "direction").all():
            output["actual_direction"] = np.where(output["y_true"] >= 0.5, 1, -1)
            output["predicted_direction"] = np.where(output["y_pred"] >= 0.5, 1, -1)
        else:
            reference = pd.to_numeric(output["prediction_reference"], errors="coerce")
            output["actual_direction"] = np.sign(pd.to_numeric(output["y_true"], errors="coerce") - reference)
            output["predicted_direction"] = np.sign(pd.to_numeric(output["y_pred"], errors="coerce") - reference)
        output["notes"] = notes
        return output[PREDICTION_COLUMNS].reset_index(drop=True)

    def _write_prediction_outputs(self) -> None:
        if self.output_dir is None:
            return
        self.predictions.to_csv(self.output_dir / "predictions" / "predictions.csv", index=False)
        self.baseline_predictions.to_csv(self.output_dir / "predictions" / "baseline_predictions.csv", index=False)
        self.model_predictions.to_csv(self.output_dir / "predictions" / "model_predictions.csv", index=False)

    def _write_metrics_outputs(self) -> None:
        if self.output_dir is None:
            return
        self.metrics.to_csv(self.output_dir / "metrics" / "metrics.csv", index=False)
        self._write_json(self.output_dir / "metrics" / "metrics_summary.json", self.metrics_engine.summarize(self.metrics))

    def _write_empty_outputs_if_needed(self) -> None:
        if self.output_dir is None:
            return
        self._finalize_predictions()
        self._write_prediction_outputs()
        if self.metrics.empty:
            self.metrics = pd.DataFrame(columns=METRICS_COLUMNS)
        self._write_metrics_outputs()
        if self.config and not (self.output_dir / "config" / "resolved_config.yaml").exists():
            try:
                self.save_resolved_config()
            except Exception:
                pass

    def _finalize_predictions(self) -> None:
        self.model_predictions = self._ensure_columns(self.model_predictions, PREDICTION_COLUMNS)
        self.baseline_predictions = self._ensure_columns(self.baseline_predictions, PREDICTION_COLUMNS)
        self.predictions = self._concat_frames([self.model_predictions, self.baseline_predictions], PREDICTION_COLUMNS)

    def _manifest_payload(self, *, status: str, errors: list[dict[str, Any]]) -> dict[str, Any]:
        output_dir = str(self.output_dir) if self.output_dir else ""
        warnings = list(dict.fromkeys([*self.warnings, *self._missing_artifact_warnings()]))
        return {
            "experiment_id": self._experiment_id(),
            "experiment_name": self._get_path(("experiment", "name")),
            "phase": self._get_path(("experiment", "phase")),
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": status,
            "config_path": str(self.config_path),
            "output_dir": output_dir,
            "git_commit": self._git_commit(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "provider": self._get_path(("data", "provider")),
            "frequency": self._get_path(("data", "frequency")),
            "universe": self._get_path(("data", "universe")) or [],
            "date_window": {
                "start_date": self._get_path(("data", "start_date")),
                "end_date": self._get_path(("data", "end_date")),
                "train_start": self._get_path(("evaluation", "train_start")),
                "train_end": self._get_path(("evaluation", "train_end")),
                "test_start": self._get_path(("evaluation", "test_start")),
                "test_end": self._get_path(("evaluation", "test_end")),
            },
            "horizons": self._get_path(("target", "horizons")) or [],
            "models": self._get_path(("models", "include")) or [],
            "baselines": self._get_path(("baselines", "include")) or [],
            "metrics": self._get_path(("evaluation", "metrics")) or [],
            "artifacts": self._artifact_index(),
            "errors": errors,
            "warnings": warnings,
            "provider_environment": self.provider_environment,
        }

    def _artifact_index(self) -> dict[str, Any]:
        if self.output_dir is None:
            return {}
        paths = {
            "original_config": self.output_dir / "config" / "original_config.yaml",
            "resolved_config": self.output_dir / "config" / "resolved_config.yaml",
            "run_manifest": self.output_dir / "manifests" / "run_manifest.json",
            "run_log": self.output_dir / "logs" / "run.log",
            "errors_log": self.output_dir / "logs" / "errors.log",
            "metrics": self.output_dir / "metrics" / "metrics.csv",
            "metrics_summary": self.output_dir / "metrics" / "metrics_summary.json",
            "predictions": self.output_dir / "predictions" / "predictions.csv",
            "baseline_predictions": self.output_dir / "predictions" / "baseline_predictions.csv",
            "model_predictions": self.output_dir / "predictions" / "model_predictions.csv",
            "summary": self.output_dir / "reports" / "summary.md",
        }
        artifacts = {
            name: {"path": str(path), "exists": path.exists()}
            for name, path in paths.items()
        }
        for name in EXPECTED_ARTIFACTS:
            path = self.output_dir / "artifacts" / name
            artifacts[name] = {"path": str(path), "exists": path.exists(), "expected": True}
        return artifacts

    def _missing_expected_artifacts(self) -> list[str]:
        if self.output_dir is None:
            return EXPECTED_ARTIFACTS
        return [name for name in EXPECTED_ARTIFACTS if not (self.output_dir / "artifacts" / name).exists()]

    def _missing_artifact_warnings(self) -> list[str]:
        return [f"expected_artifact_not_generated:{name}" for name in self._missing_expected_artifacts()]

    def _status_from_errors(self) -> str:
        if self.errors and self.predictions.empty:
            return "failed"
        if self.errors:
            return "completed_with_errors"
        return "completed"

    def _record_error(
        self,
        *,
        stage: str,
        message: str,
        ticker: str | None = None,
        horizon: int | None = None,
        model_name: str | None = None,
        traceback_text: str | None = None,
    ) -> None:
        error = {
            "stage": stage,
            "message": message,
            "ticker": ticker,
            "horizon": horizon,
            "model_name": model_name,
            "traceback": traceback_text,
        }
        if self.provider_environment:
            error["provider_environment"] = self.provider_environment
        self.errors.append(error)
        if self.output_dir is not None:
            self._log("errors.log", json.dumps({k: v for k, v in error.items() if k != "traceback"}, default=str))
            if traceback_text:
                self._log("errors.log", traceback_text)

    def _log(self, relative_name: str, message: str) -> None:
        if self.output_dir is None:
            return
        path = self.output_dir / "logs" / relative_name
        timestamp = datetime.now(UTC).isoformat()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")

    def _initialize_logs(self) -> None:
        if self.output_dir is None:
            return
        for name in ("run.log", "errors.log"):
            (self.output_dir / "logs" / name).write_text("", encoding="utf-8")

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")

    @staticmethod
    def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=columns)
        result = frame.copy()
        for column in columns:
            if column not in result.columns:
                result[column] = None
        return result[columns].reset_index(drop=True)

    @staticmethod
    def _concat_frames(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
        clean = [frame for frame in frames if frame is not None and not frame.empty]
        if not clean:
            return pd.DataFrame(columns=columns)
        result = pd.concat(clean, ignore_index=True)
        for column in columns:
            if column not in result.columns:
                result[column] = None
        return result[columns].reset_index(drop=True)

    @staticmethod
    def _append_frame(existing: pd.DataFrame, addition: pd.DataFrame) -> pd.DataFrame:
        if addition is None or addition.empty:
            return existing
        if existing is None or existing.empty:
            return addition.reset_index(drop=True)
        return pd.concat([existing, addition], ignore_index=True)

    @staticmethod
    def _validate_ohlcv_frame(frame: pd.DataFrame, ticker: str) -> None:
        missing = REQUIRED_OHLCV_SCHEMA - set(frame.columns)
        if missing:
            raise ValueError(f"OHLCV data for {ticker} is missing required columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"OHLCV data for {ticker} is empty")
        parsed_dates = pd.to_datetime(frame["date"], errors="coerce")
        if parsed_dates.isna().any():
            raise ValueError(f"OHLCV data for {ticker} contains unparseable dates")
        for column in ("open", "high", "low", "close", "volume"):
            if pd.to_numeric(frame[column], errors="coerce").isna().any():
                raise ValueError(f"OHLCV data for {ticker} contains non-numeric {column} values")

    def _model_params(self, model_name: str) -> dict[str, Any]:
        params = dict((self._get_path(("models", "params")) or {}).get(model_name, {}))
        return params

    def _check_provider_import(self, provider_name: str) -> dict[str, Any]:
        """Record active interpreter details for provider import diagnostics."""
        details: dict[str, Any] = {
            "provider_name": provider_name,
            "python_executable": sys.executable,
            "python_version": sys.version,
            "import_status": "unknown",
            "import_error_message": None,
            "module_file": None,
        }
        try:
            module = import_module(provider_name)
        except Exception as exc:
            details["import_status"] = "failed"
            details["import_error_message"] = str(exc)
        else:
            details["import_status"] = "available"
            details["module_file"] = getattr(module, "__file__", None)
        self.provider_environment = details
        return details

    def _new_run_id(self) -> str:
        return f"{self._experiment_id()}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"

    def _experiment_id(self) -> str:
        return str(self._get_path(("experiment", "id")) or "UNKNOWN-EXPERIMENT")

    def _get_path(self, path: tuple[str, ...]) -> Any:
        current: Any = self.config
        for part in path:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    @staticmethod
    def _path_label(path: tuple[str, ...]) -> str:
        return ".".join(path)

    @staticmethod
    def _parse_iso_date(value: str, label: str) -> date:
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be a valid ISO date") from exc

    @staticmethod
    def _resolve_repo_path(value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve()

    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return path == parent

    @staticmethod
    def _git_commit() -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
