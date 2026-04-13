"""Optuna-based tuning for risk/regime/allocation parameters using validation data only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.benchmark.evaluator import MetricsEvaluator
from src.ml.benchmark.system_benchmark import BenchmarkModeSpec, SystemBenchmarkRunner
from src.ml.trainer import DualModelTrainer, HORIZON_DAYS, SEQUENCE_ALGORITHMS

try:
    import optuna

    HAS_OPTUNA = True
except ImportError:  # pragma: no cover
    HAS_OPTUNA = False


@dataclass(frozen=True)
class ObjectiveWeights:
    sharpe: float = 1.0
    sortino: float = 0.5
    max_drawdown: float = 0.5
    turnover: float = 0.1


class RiskTuningRunner:
    """Tune full-system risk controls using validation-only strategy metrics."""

    def __init__(
        self,
        *,
        model_root: str | Path,
        evaluator: MetricsEvaluator | None = None,
        objective_weights: ObjectiveWeights | None = None,
        fee: float = 0.0015,
        slippage: float = 0.002,
    ) -> None:
        self.model_root = Path(model_root)
        self.evaluator = evaluator or MetricsEvaluator()
        self.objective_weights = objective_weights or ObjectiveWeights()
        self.eval_config = {"fee": fee, "slippage": slippage}
        self.signal_builder = SystemBenchmarkRunner(model_root=model_root, evaluator=self.evaluator, fee=fee, slippage=slippage)

    @staticmethod
    def _full_system_mode(risk_config: dict[str, Any]) -> BenchmarkModeSpec:
        return BenchmarkModeSpec(
            name="full_system",
            description="Risk/regime/allocation system under tuning.",
            risk_config=risk_config,
        )

    def _candidate_from_trial(self, trial: Any) -> dict[str, Any]:
        return {
            "risk_enabled": True,
            "enable_covar": True,
            "enable_risk_engine": True,
            "enable_regime_detection": True,
            "enable_regime_switching": True,
            "enable_risk_allocation": True,
            "covar_quantile": trial.suggest_float("covar_quantile", 0.01, 0.10),
            "covar_window": trial.suggest_int("covar_window", 20, 120),
            "risk_penalty_strength": trial.suggest_float("risk_penalty_strength", 0.1, 3.0),
            "high_vol_threshold": trial.suggest_float("high_vol_threshold", 0.015, 0.08),
            "crisis_drawdown_threshold": trial.suggest_float("crisis_drawdown_threshold", -0.25, -0.05),
            "crisis_delta_covar_threshold": trial.suggest_float("crisis_delta_covar_threshold", 0.005, 0.05),
            "high_vol_exposure_cut": trial.suggest_float("high_vol_exposure_cut", 0.3, 0.9),
            "crisis_exposure_cut": trial.suggest_float("crisis_exposure_cut", 0.05, 0.5),
            "regime_method": "threshold",
            "random_seed": 42,
            "simulations": 10000,
            "confidence_levels": [0.95, 0.99],
        }

    def _score(self, metrics: dict[str, float], trade_stats: dict[str, float]) -> float:
        return float(
            self.objective_weights.sharpe * metrics["sharpe"]
            + self.objective_weights.sortino * metrics["sortino"]
            - self.objective_weights.max_drawdown * abs(metrics["max_drawdown"])
            - self.objective_weights.turnover * trade_stats["turnover"]
        )

    def _evaluate_candidate(
        self,
        *,
        files: list[Path],
        algorithms: list[str],
        output_root: Path,
        risk_config: dict[str, Any],
        primary_algorithm: str | None,
        horizons: list[str] | None,
        sequence_length: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        patience: int,
        max_depth: int | None,
        min_samples_split: int,
        min_samples_leaf: int,
        criterion: str | None,
    ) -> tuple[float, list[dict[str, Any]]]:
        mode = self._full_system_mode(risk_config)
        rows: list[dict[str, Any]] = []
        for csv_path in files:
            ticker = csv_path.stem.upper()
            df = pd.read_csv(csv_path)
            trainer = DualModelTrainer(model_dir=output_root / ticker)
            trainer.train(
                ticker=ticker,
                df=df,
                algorithms=algorithms,
                primary_algorithm=primary_algorithm,
                horizons=horizons,
                sequence_length=sequence_length,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                patience=patience,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                risk_config=risk_config,
            )

            prepared = trainer.prepare_ticker_data(
                ticker=ticker,
                df=df,
                max_sequence_length=sequence_length,
                risk_config=risk_config,
            )
            labeled = trainer._add_targets(prepared.feature_frame)
            manifest = trainer._manifests[ticker]

            for horizon, horizon_info in manifest.get("horizons", {}).items():
                for algorithm, algorithm_info in horizon_info.get("algorithms", {}).items():
                    feature_columns = algorithm_info.get("feature_columns", manifest.get("feature_columns", []))
                    algo_sequence_length = int(algorithm_info.get("sequence_length") or sequence_length)
                    problem = trainer._build_horizon_problem(
                        labeled,
                        feature_columns,
                        horizon,
                        algo_sequence_length,
                    )
                    if problem is None:
                        continue
                    use_sequence = algorithm in SEQUENCE_ALGORITHMS
                    inputs = problem["sequence" if use_sequence else "tabular"]
                    x_val = inputs["X_val"]
                    if len(x_val) == 0:
                        continue

                    trend_model = trainer._get_loaded_model(ticker, algorithm, horizon, "trend")
                    predicted_direction = np.asarray(trend_model.predict(x_val), dtype=int)
                    val_frame = inputs["val_feature_frame"].reset_index(drop=True)
                    signal = self.signal_builder._build_signal(mode, ticker, predicted_direction, val_frame)
                    evaluation = DualModelTrainer.evaluate_strategy_for_horizon(
                        signal,
                        np.asarray(inputs["y_val_return"], dtype=float),
                        HORIZON_DAYS[horizon],
                        evaluator=self.evaluator,
                        config=self.eval_config,
                    )
                    score = self._score(evaluation["metrics"], evaluation["trade_stats"])
                    rows.append(
                        {
                            "ticker": ticker,
                            "horizon": horizon,
                            "algorithm": algorithm,
                            "validation_rows": int(len(x_val)),
                            "score": score,
                            "sharpe": evaluation["metrics"]["sharpe"],
                            "sortino": evaluation["metrics"]["sortino"],
                            "max_drawdown": evaluation["metrics"]["max_drawdown"],
                            "turnover": evaluation["trade_stats"]["turnover"],
                        }
                    )

        if not rows:
            return float("-inf"), []
        score = float(np.mean([row["score"] for row in rows]))
        return score, rows

    def run(
        self,
        *,
        files: list[Path],
        algorithms: list[str],
        output_root: str | Path,
        report_path: str | Path,
        max_trials: int = 10,
        primary_algorithm: str | None = None,
        horizons: list[str] | None = None,
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 30,
        patience: int = 5,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str | None = None,
    ) -> dict[str, Any]:
        output_root = Path(output_root)
        report_path = Path(report_path)
        trial_rows: list[dict[str, Any]] = []

        def _run_candidate(candidate_config: dict[str, Any], trial_number: int) -> float:
            score, rows = self._evaluate_candidate(
                files=files,
                algorithms=algorithms,
                output_root=output_root / "trials" / f"trial_{trial_number}",
                risk_config=candidate_config,
                primary_algorithm=primary_algorithm,
                horizons=horizons,
                sequence_length=sequence_length,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                patience=patience,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
            )
            trial_rows.append({"trial_number": trial_number, "score": score, **candidate_config})
            return score

        if HAS_OPTUNA:
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction="maximize", sampler=sampler)

            def objective(trial: Any) -> float:
                candidate_config = self._candidate_from_trial(trial)
                return _run_candidate(candidate_config, int(trial.number))

            study.optimize(objective, n_trials=max_trials)
            best_params = {
                "risk_enabled": True,
                "enable_covar": True,
                "enable_risk_engine": True,
                "enable_regime_detection": True,
                "enable_regime_switching": True,
                "enable_risk_allocation": True,
                "regime_method": "threshold",
                "random_seed": 42,
                "simulations": 10000,
                "confidence_levels": [0.95, 0.99],
                **study.best_params,
            }
            best_score = float(study.best_value)
        else:
            best_params = {
                "risk_enabled": True,
                "enable_covar": True,
                "enable_risk_engine": True,
                "enable_regime_detection": True,
                "enable_regime_switching": True,
                "enable_risk_allocation": True,
                "covar_quantile": 0.05,
                "covar_window": 60,
                "risk_penalty_strength": 1.0,
                "high_vol_threshold": 0.03,
                "crisis_drawdown_threshold": -0.12,
                "crisis_delta_covar_threshold": 0.015,
                "high_vol_exposure_cut": 0.6,
                "crisis_exposure_cut": 0.25,
                "regime_method": "threshold",
                "random_seed": 42,
                "simulations": 10000,
                "confidence_levels": [0.95, 0.99],
            }
            best_score = _run_candidate(best_params, 0)

        best_model_root = output_root / "best_models"
        for csv_path in files:
            ticker = csv_path.stem.upper()
            df = pd.read_csv(csv_path)
            trainer = DualModelTrainer(model_dir=best_model_root)
            trainer.train(
                ticker=ticker,
                df=df,
                algorithms=algorithms,
                primary_algorithm=primary_algorithm,
                horizons=horizons,
                sequence_length=sequence_length,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                patience=patience,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                risk_config=best_params,
            )

        trials_df = pd.DataFrame(trial_rows).sort_values("score", ascending=False).reset_index(drop=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path = report_path if report_path.suffix.lower() == ".csv" else report_path.with_suffix(".csv")
        json_path = csv_path.with_suffix(".json")
        markdown_path = Path("reports") / "risk_tuning_report.md"
        trials_df.to_csv(csv_path, index=False)
        json_path.write_text(
            json.dumps(
                {
                    "best_score": best_score,
                    "best_params": best_params,
                    "trials": trials_df.to_dict(orient="records"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            "\n".join(
                [
                    "# Risk Tuning Report",
                    "",
                    f"Best validation score: `{best_score:.6f}`",
                    "",
                    "## Best Parameters",
                    *[f"- `{key}`: `{value}`" for key, value in best_params.items()],
                    "",
                    "## Trial Leaderboard",
                    trials_df.head(10).to_string(index=False) if not trials_df.empty else "_No trials executed._",
                ]
            ),
            encoding="utf-8",
        )

        return {
            "best_score": best_score,
            "best_params": best_params,
            "trials": trials_df,
            "csv_path": csv_path,
            "json_path": json_path,
            "markdown_path": markdown_path,
            "best_model_root": best_model_root,
        }
