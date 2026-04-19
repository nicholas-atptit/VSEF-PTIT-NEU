"""Quant-core orchestration helpers built on the existing evaluation stack."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.core.model_governance import get_run_mode_spec
from src.ensemble.weighted import WeightedEnsembleModel
from src.evaluation.backtest import BacktestConfig, CostAwareBacktester
from src.evaluation.forecast_rehab import create_rehab_forecast_model, forecast_rehab_policy_baseline
from src.evaluation.hardening import DEFAULT_TICKER_GROUPS
from src.evaluation.targets import ForecastTargetSpec, build_target_spec
from src.evaluation.walkforward import WalkForwardConfig, WalkForwardEvaluator, summarize_forecasts
from src.forecast.registry import forecast_model_governance_table, resolve_forecast_model_registrations
from src.regime.markov_switching import MarkovSwitchingRegimeModel
from src.risk.drawdown import DrawdownRiskModel
from src.risk.garch import GARCHRiskModel
from src.risk.var_cvar import VaRCVaRRiskModel
from src.strategy.execution_policy import PolicyConfiguration, execute_policy_configuration


PRESET_CONFIGS: dict[str, dict[str, Any]] = {
    "smoke": {
        "group_names": ["small_banks"],
        "horizons": [5],
        "target_names": ["forward_return"],
        "evaluation_config": {
            "train_size": 252,
            "test_size": 21,
            "step_size": 21,
            "gap_size": 0,
            "max_windows": 1,
        },
    },
    "medium": {
        "group_names": ["small_banks", "mixed_large_cap"],
        "horizons": [5, 10],
        "target_names": ["forward_return", "direction_binary"],
        "evaluation_config": {
            "train_size": 252,
            "test_size": 21,
            "step_size": 21,
            "gap_size": 0,
            "max_windows": 2,
        },
    },
    "full_forecast_daily": {
        "group_names": ["small_banks", "mixed_large_cap", "vn100_subset"],
        "horizons": [1, 5, 10],
        "target_names": ["forward_return", "forward_log_return", "direction_binary"],
        "evaluation_config": {
            "train_size": 252,
            "test_size": 21,
            "step_size": 21,
            "gap_size": 0,
            "max_windows": 2,
        },
    },
}


def build_quant_core_matrix_config(
    preset: str = "medium",
    *,
    group_names: list[str] | None = None,
    ticker_groups: dict[str, list[str]] | None = None,
    horizons: list[int] | None = None,
    target_names: list[str] | None = None,
) -> dict[str, Any]:
    """Build the scenario matrix used by the quant-core runner."""

    key = str(preset or "medium").strip().lower()
    if key not in PRESET_CONFIGS:
        raise ValueError(f"Unsupported quant-core preset '{preset}'")

    all_groups = dict(DEFAULT_TICKER_GROUPS)
    all_groups.update(
        {
            str(name): [str(ticker).upper() for ticker in tickers]
            for name, tickers in dict(ticker_groups or {}).items()
        }
    )
    preset_config = PRESET_CONFIGS[key]
    selected_group_names = list(group_names or preset_config["group_names"])
    selected_groups: list[dict[str, Any]] = []
    for group_name in selected_group_names:
        if group_name not in all_groups:
            raise ValueError(f"Ticker group '{group_name}' is not defined")
        selected_groups.append(
            {
                "group_name": str(group_name),
                "tickers": list(all_groups[group_name]),
            }
        )

    resolved_horizons = [int(value) for value in (horizons or preset_config["horizons"])]
    resolved_targets = [str(value).strip().lower() for value in (target_names or preset_config["target_names"])]
    return {
        "preset": key,
        "ticker_groups": selected_groups,
        "horizons": resolved_horizons,
        "target_names": resolved_targets,
        "evaluation_config": dict(preset_config["evaluation_config"]),
        "policy_baseline": forecast_rehab_policy_baseline(),
    }


def build_quant_core_core_frame(matrix_config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    evaluation_config = dict(matrix_config.get("evaluation_config", {}))
    for group in matrix_config.get("ticker_groups", []):
        for horizon in matrix_config.get("horizons", []):
            for target_name in matrix_config.get("target_names", []):
                target_spec = build_target_spec(target_name)
                rows.append(
                    {
                        "core_run_id": f"{group['group_name']}_h{int(horizon):02d}_{target_spec.name}",
                        "preset": str(matrix_config.get("preset", "medium")),
                        "group_name": str(group["group_name"]),
                        "tickers": list(group["tickers"]),
                        "ticker_count": len(group["tickers"]),
                        "horizon": int(horizon),
                        "target_name": target_spec.name,
                        "target_type": target_spec.target_type,
                        "target_column": target_spec.target_column,
                        "target_family": target_spec.target_family,
                        "target_tradable": bool(target_spec.tradable_output),
                        **evaluation_config,
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "core_run_id",
                "preset",
                "group_name",
                "tickers",
                "ticker_count",
                "horizon",
                "target_name",
                "target_type",
                "target_column",
                "target_family",
                "target_tradable",
                "train_size",
                "test_size",
                "step_size",
                "gap_size",
                "max_windows",
            ]
        )
    return pd.DataFrame(rows).sort_values(["group_name", "horizon", "target_name"]).reset_index(drop=True)


def _normalize_metadata(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    values = row if isinstance(row, dict) else row.to_dict()
    metadata = dict(values)
    tickers = metadata.get("tickers", [])
    metadata["ticker_group_members"] = ",".join(str(ticker) for ticker in tickers)
    for key, value in list(metadata.items()):
        if isinstance(value, (list, tuple)):
            metadata[key] = ",".join(str(item) for item in value)
    return metadata


def _annotate(frame: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    annotated = frame.copy()
    for key, value in metadata.items():
        if key == "tickers":
            continue
        annotated[key] = value
    return annotated


def _annotate_model_context(frame: pd.DataFrame, model_context: dict[str, Any]) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["model_family"] = str(model_context["family"])
    annotated["model_role"] = str(model_context["role"])
    annotated["model_status"] = str(model_context["status"])
    annotated["research_priority"] = int(model_context["research_priority"])
    annotated["supports_policy_eval"] = bool(model_context["supports_policy_eval"])
    annotated["model_notes"] = str(model_context["notes"])
    return annotated


def _ensemble_context() -> dict[str, Any]:
    return {
        "family": "ensemble",
        "role": "ensemble",
        "status": "derived",
        "research_priority": 0,
        "supports_policy_eval": True,
        "notes": "Derived weighted ensemble from evaluated governed forecasts.",
    }


def _drawdown_state(current_drawdown: float, *, elevated: float = -0.05, severe: float = -0.10) -> str:
    if float(current_drawdown) <= float(severe):
        return "severe"
    if float(current_drawdown) <= float(elevated):
        return "elevated"
    return "normal"


def _threshold_regime_frame(
    history_df: pd.DataFrame,
    *,
    lookback: int,
    bull_threshold: float,
    bear_threshold: float,
) -> pd.DataFrame:
    prepared = history_df.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
    if "daily_return" not in prepared.columns and "close" in prepared.columns:
        prepared["daily_return"] = pd.to_numeric(prepared["close"], errors="coerce").pct_change()
    returns = pd.to_numeric(prepared["daily_return"], errors="coerce").fillna(0.0)
    compounded = (1.0 + returns).rolling(window=int(lookback), min_periods=1).apply(np.prod, raw=True) - 1.0
    labels = pd.Series("sideway", index=prepared.index, dtype="object")
    labels.loc[compounded > float(bull_threshold)] = "bull"
    labels.loc[compounded < float(bear_threshold)] = "bear"
    return pd.DataFrame(
        {
            "timestamp": prepared["timestamp"],
            "ticker": prepared["ticker"].astype(str).str.upper(),
            "regime_label": labels,
            "regime_prob_bull": (labels == "bull").astype(float),
            "regime_prob_bear": (labels == "bear").astype(float),
            "regime_prob_sideway": (labels == "sideway").astype(float),
            "source_model": "markov_switching_threshold_fallback",
            "window_id": prepared.get("window_id", pd.Series("unassigned", index=prepared.index)).astype(str),
        }
    )


def build_window_garch_risk_frame(
    datasets: dict[str, Any],
    window_summary: pd.DataFrame,
    *,
    horizon: int,
) -> pd.DataFrame:
    risk_rows: list[dict[str, Any]] = []
    for row in window_summary.itertuples(index=False):
        dataset = datasets[str(row.ticker)]
        frame = dataset.frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        train_df = frame[
            (frame["timestamp"] >= pd.Timestamp(row.train_start))
            & (frame["timestamp"] <= pd.Timestamp(row.train_end))
        ].copy()
        test_df = frame[
            (frame["timestamp"] >= pd.Timestamp(row.test_start))
            & (frame["timestamp"] <= pd.Timestamp(row.test_end))
        ].copy()
        returns = pd.to_numeric(train_df["daily_return"], errors="coerce").dropna()
        if returns.empty or test_df.empty:
            continue

        try:
            metrics = GARCHRiskModel().fit(returns).forecast_risk(horizon=horizon)
        except Exception:
            fallback = VaRCVaRRiskModel().fit(returns).forecast_risk(horizon=horizon)
            drawdown = DrawdownRiskModel().fit(returns).forecast_risk(horizon=horizon)
            metrics = {
                **fallback,
                **drawdown,
                "risk_model": "var_cvar_drawdown_fallback",
                "distribution": "historical_fallback",
                "vol_forecast": float(fallback.get("volatility", 0.0)),
                "drawdown_state": _drawdown_state(float(drawdown.get("current_drawdown", 0.0))),
            }
        for timestamp in pd.to_datetime(test_df["timestamp"], errors="coerce").dropna():
            risk_rows.append(
                {
                    "timestamp": timestamp,
                    "ticker": str(row.ticker),
                    "window_id": str(row.window_id),
                    "source_model": metrics["risk_model"],
                    **metrics,
                }
            )
    return pd.DataFrame(risk_rows).sort_values(["timestamp", "ticker", "window_id"]).reset_index(drop=True)


def build_window_regime_frame(
    datasets: dict[str, Any],
    window_summary: pd.DataFrame,
    *,
    lookback: int,
    bull_threshold: float,
    bear_threshold: float,
) -> pd.DataFrame:
    regime_frames: list[pd.DataFrame] = []
    for row in window_summary.itertuples(index=False):
        dataset = datasets[str(row.ticker)]
        frame = dataset.frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        train_df = frame[
            (frame["timestamp"] >= pd.Timestamp(row.train_start))
            & (frame["timestamp"] <= pd.Timestamp(row.train_end))
        ].copy()
        history_df = frame[
            (frame["timestamp"] >= pd.Timestamp(row.train_start))
            & (frame["timestamp"] <= pd.Timestamp(row.test_end))
        ].copy()
        test_df = frame[
            (frame["timestamp"] >= pd.Timestamp(row.test_start))
            & (frame["timestamp"] <= pd.Timestamp(row.test_end))
        ].copy()
        if train_df.empty or history_df.empty or test_df.empty:
            continue
        train_df["window_id"] = str(row.window_id)
        history_df["window_id"] = str(row.window_id)
        try:
            model = MarkovSwitchingRegimeModel(
                return_column="daily_return",
                lookback=lookback,
                bull_threshold=bull_threshold,
                bear_threshold=bear_threshold,
                min_train_observations=max(80, min(len(train_df), 120)),
            ).fit(train_df, config={"window_id": str(row.window_id)})
            regime_history = model.predict(history_df)
        except Exception:
            regime_history = _threshold_regime_frame(
                history_df,
                lookback=lookback,
                bull_threshold=bull_threshold,
                bear_threshold=bear_threshold,
            )
        regime_frames.append(
            regime_history[regime_history["timestamp"].isin(pd.to_datetime(test_df["timestamp"], errors="coerce"))].copy()
        )
    if not regime_frames:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "ticker",
                "regime_label",
                "regime_prob_bull",
                "regime_prob_bear",
                "regime_prob_sideway",
                "source_model",
                "window_id",
            ]
        )
    return pd.concat(regime_frames, ignore_index=True).sort_values(["timestamp", "ticker", "window_id"]).reset_index(drop=True)


def build_market_data(datasets: dict[str, Any]) -> dict[str, pd.DataFrame]:
    return {
        ticker: dataset.frame[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        for ticker, dataset in datasets.items()
    }


def build_quant_core_policy_configuration(policy_baseline: dict[str, Any]) -> PolicyConfiguration:
    return PolicyConfiguration(
        policy_variant=str(policy_baseline["policy_variant"]),
        strategy_variant="forecast_plus_risk_and_regime",
        policy_label=str(policy_baseline["policy_label"]),
        threshold_policy=str(policy_baseline["threshold_policy"]),
        sizing_profile=str(policy_baseline["sizing_profile"]),
        sizing_label=str(policy_baseline["sizing_profile"]),
        use_risk_context=bool(policy_baseline["use_risk_context"]),
        use_regime_context=bool(policy_baseline["use_regime_context"]),
        use_volatility_sizing=bool(policy_baseline["use_volatility_sizing"]),
        use_drawdown_control=bool(policy_baseline["use_drawdown_control"]),
        use_regime_sizing=bool(policy_baseline["use_regime_sizing"]),
        sizing_mode=str(policy_baseline["sizing_mode"]),
        fixed_position_size=policy_baseline.get("fixed_position_size"),
        min_position_size=float(policy_baseline["min_position_size"]),
        max_position_size=float(policy_baseline["max_position_size"]),
        volatility_target_scale=float(policy_baseline["volatility_target_scale"]),
        drawdown_haircut_strength=float(policy_baseline["drawdown_haircut_strength"]),
        regime_multiplier_strength=float(policy_baseline["regime_multiplier_strength"]),
        policy_family="phase26_default_candidate",
        ablation_labels=("QUANT_CORE_DEFAULT_POLICY",),
    )


def evaluate_governed_forecasts(
    evaluator: WalkForwardEvaluator,
    *,
    target_spec: ForecastTargetSpec,
    run_mode: str,
    requested_model_names: list[str] | None = None,
    requested_roles: list[str] | None = None,
    include_ensemble: bool = True,
) -> dict[str, Any]:
    registrations = resolve_forecast_model_registrations(
        run_mode=run_mode,
        roles=requested_roles,
        model_names=requested_model_names,
        target_type=target_spec.name,
    )
    if not registrations:
        raise RuntimeError("No governed forecast models were available for the requested run mode and filters")

    forecast_frames: list[pd.DataFrame] = []
    evaluated_models: list[str] = []
    skipped_models: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    datasets: dict[str, Any] | None = None
    window_summary: pd.DataFrame | None = None

    for registration in registrations:
        model_name = registration.governance.model_name
        model_context = registration.governance.to_dict()
        try:
            result = evaluator.evaluate([create_rehab_forecast_model(model_name, target_spec=target_spec)])
        except Exception as exc:
            skipped_models.append(
                {
                    "model_name": model_name,
                    "reason": str(exc),
                    "role": registration.governance.role,
                    "status": registration.governance.status,
                }
            )
            execution_rows.append(
                {
                    "model_name": model_name,
                    "model_family": registration.governance.family,
                    "model_role": registration.governance.role,
                    "model_status": registration.governance.status,
                    "run_success": False,
                    "warning_count": 1,
                    "missing_output_count": 1,
                    "failure_reason": str(exc),
                }
            )
            continue

        forecasts = _annotate_model_context(result["forecasts"], model_context)
        forecast_frames.append(forecasts)
        evaluated_models.append(model_name)
        if datasets is None:
            datasets = result["datasets"]
        if window_summary is None:
            window_summary = result["window_summary"][
                ["ticker", "window_id", "train_start", "train_end", "test_start", "test_end"]
            ].drop_duplicates()
        execution_rows.append(
            {
                "model_name": model_name,
                "model_family": registration.governance.family,
                "model_role": registration.governance.role,
                "model_status": registration.governance.status,
                "run_success": True,
                "warning_count": 0,
                "missing_output_count": 0,
                "failure_reason": "",
            }
        )

    if not forecast_frames or datasets is None or window_summary is None:
        raise RuntimeError("No governed forecast models completed successfully")

    forecast_df = pd.concat(forecast_frames, ignore_index=True).sort_values(
        ["timestamp", "ticker", "research_priority", "model_name"]
    ).reset_index(drop=True)
    forecast_summary = summarize_forecasts(forecast_df)

    governance_frame = forecast_model_governance_table(
        run_mode=run_mode,
        roles=requested_roles,
        model_names=requested_model_names,
        target_type=target_spec.name,
        include_parked=False,
    )
    if not governance_frame.empty:
        forecast_summary = forecast_summary.merge(
            governance_frame.rename(
                columns={
                    "family": "model_family",
                    "role": "model_role",
                    "status": "model_status",
                    "notes": "model_notes",
                }
            )[
                [
                    "model_name",
                    "model_family",
                    "model_role",
                    "model_status",
                    "research_priority",
                    "supports_policy_eval",
                    "model_notes",
                ]
            ],
            on="model_name",
            how="left",
        )

    if include_ensemble and len(evaluated_models) > 1:
        per_model_frames = [group.copy() for _, group in forecast_df.groupby("model_name", sort=True)]
        ensemble = WeightedEnsembleModel()
        ensemble_forecast = ensemble.combine(
            per_model_frames,
            context={"forecast_summary": summarize_forecasts(forecast_df)},
        )
        ensemble_forecast = _annotate_model_context(ensemble_forecast, _ensemble_context())
        forecast_df = pd.concat([forecast_df, ensemble_forecast], ignore_index=True).sort_values(
            ["timestamp", "ticker", "research_priority", "model_name"]
        ).reset_index(drop=True)
        forecast_summary = summarize_forecasts(forecast_df).merge(
            pd.DataFrame(
                [
                    *governance_frame.rename(
                        columns={
                            "family": "model_family",
                            "role": "model_role",
                            "status": "model_status",
                            "notes": "model_notes",
                        }
                    )[
                        [
                            "model_name",
                            "model_family",
                            "model_role",
                            "model_status",
                            "research_priority",
                            "supports_policy_eval",
                            "model_notes",
                        ]
                    ].to_dict(orient="records"),
                    {
                        "model_name": "weighted_ensemble",
                        "model_family": "ensemble",
                        "model_role": "ensemble",
                        "model_status": "derived",
                        "research_priority": 0,
                        "supports_policy_eval": True,
                        "model_notes": _ensemble_context()["notes"],
                    },
                ]
            ),
            on="model_name",
            how="left",
        )
        evaluated_models.append("weighted_ensemble")
        execution_rows.append(
            {
                "model_name": "weighted_ensemble",
                "model_family": "ensemble",
                "model_role": "ensemble",
                "model_status": "derived",
                "run_success": True,
                "warning_count": 0,
                "missing_output_count": 0,
                "failure_reason": "",
            }
        )

    forecast_summary_by_horizon = summarize_forecasts(
        forecast_df,
        group_columns=["model_name", "horizon"],
    )
    if not governance_frame.empty:
        forecast_summary_by_horizon = forecast_summary_by_horizon.merge(
            pd.DataFrame(
                [
                    *governance_frame.rename(
                        columns={
                            "family": "model_family",
                            "role": "model_role",
                            "status": "model_status",
                            "notes": "model_notes",
                        }
                    )[
                        [
                            "model_name",
                            "model_family",
                            "model_role",
                            "model_status",
                            "research_priority",
                            "supports_policy_eval",
                            "model_notes",
                        ]
                    ].to_dict(orient="records"),
                    {
                        "model_name": "weighted_ensemble",
                        "model_family": "ensemble",
                        "model_role": "ensemble",
                        "model_status": "derived",
                        "research_priority": 0,
                        "supports_policy_eval": True,
                        "model_notes": _ensemble_context()["notes"],
                    },
                ]
            ),
            on="model_name",
            how="left",
        )

    return {
        "forecast_registrations": registrations,
        "forecasts": forecast_df,
        "forecast_summary": forecast_summary,
        "forecast_summary_by_horizon": forecast_summary_by_horizon,
        "window_summary": window_summary,
        "datasets": datasets,
        "evaluated_models": evaluated_models,
        "skipped_models": skipped_models,
        "model_execution_log": pd.DataFrame(execution_rows).sort_values(["model_family", "model_name"]).reset_index(drop=True),
    }


def run_quant_core_scenario(
    core_row: dict[str, Any],
    *,
    run_mode: str,
    requested_model_names: list[str] | None = None,
    requested_roles: list[str] | None = None,
    include_ensemble: bool = True,
    seed: int = 42,
    allow_short: bool = False,
    risk_budget: float = 0.02,
    max_position_size: float = 1.0,
    transaction_fee_bps: float = 15.0,
    slippage_bps: float = 20.0,
    regime_lookback: int = 20,
    regime_bull_threshold: float = 0.03,
    regime_bear_threshold: float = -0.03,
) -> dict[str, Any]:
    target_spec = build_target_spec(
        str(core_row["target_name"]),
        target_column=str(core_row["target_column"]),
    )
    metadata = _normalize_metadata(core_row)
    metadata["run_mode"] = get_run_mode_spec(run_mode).run_mode
    metadata["requested_model_roles"] = ",".join(str(value) for value in (requested_roles or []))
    metadata["requested_model_names"] = ",".join(str(value).lower() for value in (requested_model_names or []))

    evaluator = WalkForwardEvaluator(
        WalkForwardConfig(
            tickers=list(core_row["tickers"]),
            horizon=int(core_row["horizon"]),
            train_size=int(core_row["train_size"]),
            test_size=int(core_row["test_size"]),
            step_size=int(core_row["step_size"]),
            gap_size=int(core_row["gap_size"]),
            max_windows=int(core_row["max_windows"]),
            target_column=target_spec.target_column,
            target_type=target_spec.name,
            seed=int(seed),
        )
    )

    forecast_result = evaluate_governed_forecasts(
        evaluator,
        target_spec=target_spec,
        run_mode=run_mode,
        requested_model_names=requested_model_names,
        requested_roles=requested_roles,
        include_ensemble=include_ensemble,
    )

    forecasts = _annotate(forecast_result["forecasts"], metadata)
    forecast_summary = _annotate(forecast_result["forecast_summary"], metadata)
    forecast_summary_by_horizon = _annotate(forecast_result["forecast_summary_by_horizon"], metadata)
    window_summary = _annotate(forecast_result["window_summary"], metadata)
    execution_log = _annotate(forecast_result["model_execution_log"], metadata)

    risk_summary = _annotate(
        build_window_garch_risk_frame(
            forecast_result["datasets"],
            forecast_result["window_summary"],
            horizon=int(core_row["horizon"]),
        ),
        metadata,
    )
    regime_summary = _annotate(
        build_window_regime_frame(
            forecast_result["datasets"],
            forecast_result["window_summary"],
            lookback=int(regime_lookback),
            bull_threshold=float(regime_bull_threshold),
            bear_threshold=float(regime_bear_threshold),
        ),
        metadata,
    )

    signals = pd.DataFrame()
    positions = pd.DataFrame()
    trades = pd.DataFrame()
    strategy_metrics = pd.DataFrame()
    equity_curve = pd.DataFrame()
    policy_summary = pd.DataFrame()

    if target_spec.tradable_output:
        policy_baseline = forecast_rehab_policy_baseline()
        signal_df, position_df = execute_policy_configuration(
            forecasts,
            policy_config=build_quant_core_policy_configuration(policy_baseline),
            threshold=float(policy_baseline["threshold"]),
            allow_short=allow_short,
            risk_df=risk_summary,
            regime_df=regime_summary,
            capital_config={
                "risk_budget": float(risk_budget),
                "max_position_size": float(max_position_size),
            },
        )
        backtester = CostAwareBacktester(
            BacktestConfig(
                horizon=int(core_row["horizon"]),
                transaction_fee_bps=float(transaction_fee_bps),
                slippage_bps=float(slippage_bps),
                allow_short=allow_short,
            )
        )
        backtest_result = backtester.run(position_df, build_market_data(forecast_result["datasets"]))
        signals = _annotate(signal_df, metadata)
        positions = _annotate(position_df, metadata)
        trades = _annotate(backtest_result["trades"], metadata)
        strategy_metrics = _annotate(backtest_result["strategy_metrics"], metadata)
        equity_curve = _annotate(backtest_result["equity_curve"], metadata)
        policy_summary = strategy_metrics.copy()

    return {
        "metadata": metadata,
        "selected_model_governance": forecast_model_governance_table(
            run_mode=run_mode,
            roles=requested_roles,
            model_names=requested_model_names,
            target_type=target_spec.name,
            include_parked=False,
        ),
        "forecasts": forecasts,
        "forecast_summary": forecast_summary,
        "forecast_summary_by_horizon": forecast_summary_by_horizon,
        "window_summary": window_summary,
        "risk_summary": risk_summary,
        "regime_summary": regime_summary,
        "signals": signals,
        "positions": positions,
        "trades": trades,
        "strategy_metrics": strategy_metrics,
        "equity_curve": equity_curve,
        "policy_summary": policy_summary,
        "model_execution_log": execution_log,
        "evaluated_models": list(forecast_result["evaluated_models"]),
        "skipped_models": [
            {**metadata, **item}
            for item in forecast_result["skipped_models"]
        ],
    }
