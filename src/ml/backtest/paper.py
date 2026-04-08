"""Paper trading engine for Phase 5 event-driven simulations."""

from __future__ import annotations

import asyncio
import datetime as dt
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from config.settings import get_settings
from src.api.schemas import QualitativeAnalysis, QuantitativeSignals
from src.ml.backtest.event_driven import get_safe_rag_context, simulate_execution_cost
from src.engine.matrix import evaluate_decision_matrix
from src.engine.risk import apply_risk_constraints
from src.ml.llm.pipeline import run_qualitative_analysis
from src.ml.data_loader import generate_mock_data, load_ohlcv_from_db, load_ohlcv_from_vnstock
from src.ml.signal_generator import SignalGenerator
from src.ml.trainer import DualModelTrainer
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyProfile:
    """Stores component-level execution timing."""

    websocket_ms: float = 0.0
    ml_compute_ms: float = 0.0
    vector_db_ms: float = 0.0
    news_fetch_ms: float = 0.0
    llm_inference_ms: float = 0.0
    matrix_engine_ms: float = 0.0
    risk_engine_ms: float = 0.0
    total_latency_seconds: float = 0.0


@dataclass
class PaperTrade:
    """Represents a single long paper trade."""

    trade_id: str = ""
    ticker: str = ""
    action: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    volume: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    entry_time: str = ""
    exit_time: str = ""
    last_mark_price: float = 0.0
    latency: LatencyProfile = field(default_factory=LatencyProfile)
    decision: str = ""
    status: str = "OPEN"


class PaperTradingEngine:
    """Event-driven paper trading engine with cash, positions, and P&L tracking."""

    def __init__(
        self,
        initial_capital: float = 100_000_000.0,
        max_risk_per_trade_pct: float = 0.05,
    ) -> None:
        self._settings = get_settings()
        self._cash = initial_capital
        self._initial_capital = initial_capital
        self._max_risk_pct = max_risk_per_trade_pct
        self._trades: list[PaperTrade] = []
        self._open_positions: dict[str, PaperTrade] = {}
        self._cycle_log: list[dict[str, Any]] = []
        self._realized_pnl = 0.0

        self._trainer = DualModelTrainer()
        self._signal_generator = SignalGenerator()

    async def run_single_cycle(
        self,
        ticker: str,
        risk_tolerance: float | None = None,
        allowed_zones: list[str] | None = None,
        use_mock: bool = False,
    ) -> dict[str, Any]:
        """Run one full event-driven cycle for a ticker."""
        normalized_ticker = ticker.upper().strip()
        zones = allowed_zones or ["zone_1", "zone_2", "zone_3"]
        applied_risk = min(float(risk_tolerance or self._max_risk_pct), self._settings.max_risk_tolerance)
        total_started_at = time.perf_counter()
        latency = LatencyProfile()

        fetch_started_at = time.perf_counter()
        market_snapshot = self._fetch_market_snapshot(normalized_ticker, use_mock=use_mock)
        current_price = market_snapshot["price"]
        as_of = market_snapshot["timestamp"]
        latency.websocket_ms = (time.perf_counter() - fetch_started_at) * 1000
        self._mark_to_market(normalized_ticker, current_price)

        ml_started_at = time.perf_counter()
        quant_payload, features = await self._run_quant_pipeline(
            ticker=normalized_ticker,
            current_price=current_price,
            risk_tolerance=applied_risk,
            use_mock=use_mock,
        )
        latency.ml_compute_ms = (time.perf_counter() - ml_started_at) * 1000

        rag_started_at = time.perf_counter()
        rag_context = get_safe_rag_context(
            ticker=normalized_ticker,
            allowed_zones=zones,
            current_time=as_of,
        )
        latency.vector_db_ms = (time.perf_counter() - rag_started_at) * 1000

        news_started_at = time.perf_counter()
        news_context = self._fetch_news_context(normalized_ticker)
        latency.news_fetch_ms = (time.perf_counter() - news_started_at) * 1000

        llm_started_at = time.perf_counter()
        # v5.0 payload mapping
        tech_signals = quant_payload["technical"]["horizons"][0]
        
        llm_result = await run_qualitative_analysis(
            ticker=normalized_ticker,
            quant_data={
                "trend_probabilities": tech_signals["trend_probs"],
                "expected_range": tech_signals["expected_range"],
            },
            rag_context=rag_context,
            news_context=news_context,
            user_risk_input=applied_risk,
        )
        latency.llm_inference_ms = (time.perf_counter() - llm_started_at) * 1000

        quant_model = QuantitativeSignals(
            trend_probabilities=tech_signals["trend_probs"],
            expected_range=tech_signals["expected_range"],
            max_upside_pct=0.0, # Handled in fission later
            max_downside_pct=0.0,
            horizon="short",
            feature_set_version="v5.0",
            action_plan=quant_payload["fusion"] # Fusion contains the action now
        )
        qual_model = QualitativeAnalysis(**llm_result)

        matrix_started_at = time.perf_counter()
        decision_action, matrix_consensus = evaluate_decision_matrix(quant_model, qual_model)
        latency.matrix_engine_ms = (time.perf_counter() - matrix_started_at) * 1000

        risk_started_at = time.perf_counter()
        atr_14 = float(features["atr_14"].iloc[-1]) if "atr_14" in features.columns else current_price * 0.05
        order_payload = None
        risk_override = None
        if decision_action in ("EXECUTE_BUY", "EXECUTE_SELL"):
            order_payload, risk_override = apply_risk_constraints(
                ticker=normalized_ticker,
                action_plan=quant_model.action_plan,
                real_time_price=current_price,
                atr_14=max(atr_14, 0.01),
                applied_risk_tolerance=quant_payload["risk"]["position_size_suggestion"] or applied_risk,
            )
            if risk_override and risk_override.fomo_check_passed is False:
                decision_action = "CANCEL_ORDER"
                order_payload = None
        latency.risk_engine_ms = (time.perf_counter() - risk_started_at) * 1000

        execution = self._apply_execution(
            ticker=normalized_ticker,
            decision_action=decision_action,
            order_payload=order_payload.model_dump() if order_payload else None,
            current_price=current_price,
            as_of=as_of,
            latency=latency,
        )

        latency.total_latency_seconds = time.perf_counter() - total_started_at
        validate_latency(latency)

        cycle_result = {
            "ticker": normalized_ticker,
            "timestamp": as_of.isoformat(),
            "current_price": round(current_price, 2),
            "decision": decision_action,
            "qualitative_status": llm_result.get("analysis_status"),
            "matrix_consensus": matrix_consensus.model_dump(),
            "risk_override": risk_override.model_dump() if risk_override else None,
            "execution": execution,
            "latency_profile": {
                "websocket_ms": round(latency.websocket_ms, 1),
                "ml_compute_ms": round(latency.ml_compute_ms, 1),
                "vector_db_ms": round(latency.vector_db_ms, 1),
                "news_fetch_ms": round(latency.news_fetch_ms, 1),
                "llm_inference_ms": round(latency.llm_inference_ms, 1),
                "matrix_engine_ms": round(latency.matrix_engine_ms, 1),
                "risk_engine_ms": round(latency.risk_engine_ms, 1),
                "total_seconds": round(latency.total_latency_seconds, 3),
                "within_sla": latency.total_latency_seconds <= self._settings.latency_sla_seconds,
            },
            "portfolio": self.get_portfolio_summary(),
        }
        self._cycle_log.append(cycle_result)
        return cycle_result

    async def run_watchlist_cycle(
        self,
        tickers: list[str],
        risk_tolerance: float | None = None,
        allowed_zones: list[str] | None = None,
        use_mock: bool = False,
    ) -> dict[str, Any]:
        """Run the full event-driven cycle for a watchlist."""
        results = []
        for ticker in tickers:
            results.append(
                await self.run_single_cycle(
                    ticker=ticker,
                    risk_tolerance=risk_tolerance,
                    allowed_zones=allowed_zones,
                    use_mock=use_mock,
                )
            )
            await asyncio.sleep(0)
        return {"results": results, "portfolio": self.get_portfolio_summary()}

    def get_trade_history(self) -> list[dict[str, Any]]:
        """Return the paper trade ledger."""
        return [
            {
                "id": trade.trade_id,
                "ticker": trade.ticker,
                "decision": trade.decision,
                "action": trade.action,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "volume": trade.volume,
                "status": trade.status,
                "pnl": trade.pnl,
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "latency_s": round(trade.latency.total_latency_seconds, 3),
            }
            for trade in self._trades
        ]

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Return cash, equity, and P&L metrics."""
        market_value = sum(position.last_mark_price * position.volume for position in self._open_positions.values())
        unrealized_pnl = sum(
            (position.last_mark_price - position.entry_price) * position.volume - position.fees
            for position in self._open_positions.values()
        )
        equity = self._cash + market_value
        total_pnl = equity - self._initial_capital

        return {
            "initial_capital": round(self._initial_capital, 2),
            "cash": round(self._cash, 2),
            "market_value": round(market_value, 2),
            "equity": round(equity, 2),
            "realized_pnl": round(self._realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round((total_pnl / self._initial_capital) * 100, 2),
            "total_trades": len(self._trades),
            "open_positions": len(self._open_positions),
        }

    async def _run_quant_pipeline(
        self,
        ticker: str,
        current_price: float,
        risk_tolerance: float,
        use_mock: bool,
    ) -> tuple[Any, Any]:
        """Run the quant stack and auto-train the model if needed."""
        raw_df = self._load_ohlcv(ticker, use_mock=use_mock)
        if len(raw_df) < 100:
            raise ValueError(f"Insufficient data for {ticker}: {len(raw_df)} rows")

        features = self._trainer.compute_features_for_ticker(ticker, raw_df)

        try:
            model_output = self._trainer.predict(ticker, features)
        except FileNotFoundError:
            logger.info("paper_trade_auto_train", ticker=ticker)
            self._trainer.train(ticker=ticker, df=raw_df)
            features = self._trainer.compute_features_for_ticker(ticker, raw_df)
            model_output = self._trainer.predict(ticker, features)

        quant_payload = await self._signal_generator.generate(
            ticker=ticker,
            current_close=current_price,
            model_output=model_output,
            risk_tolerance=risk_tolerance,
        )
        return quant_payload, features

    @staticmethod
    def _load_ohlcv(ticker: str, use_mock: bool = False) -> Any:
        """Load OHLCV data with DB -> vnstock -> mock fallback."""
        if use_mock:
            return generate_mock_data(ticker=ticker)

        try:
            return load_ohlcv_from_db(ticker)
        except Exception:
            try:
                return load_ohlcv_from_vnstock(ticker)
            except Exception:
                logger.warning("paper_trade_mock_fallback", ticker=ticker)
                return generate_mock_data(ticker=ticker)

    def _fetch_market_snapshot(self, ticker: str, use_mock: bool = False) -> dict[str, Any]:
        """Fetch the latest trade price for a ticker."""
        if use_mock:
            df = generate_mock_data(ticker=ticker)
            return {
                "price": float(df["close"].iloc[-1]),
                "timestamp": dt.datetime.now(dt.UTC),
            }

        try:
            import os
            from vnstock import Vnstock

            settings = get_settings()
            os.environ["VNAI_API_KEY"] = settings.vnstock_api_key
            os.environ["VNSTOCK_API_KEY"] = settings.vnstock_api_key

            stock = Vnstock().stock(symbol=ticker, source="VCI")
            intraday = stock.quote.intraday()
            if intraday is not None and not intraday.empty:
                latest = intraday.iloc[-1]
                price = float(latest.get("price", latest.get("close", 0.0)))
                timestamp = self._normalize_timestamp(latest.get("time"))
                if price > 0:
                    return {"price": price, "timestamp": timestamp}

            history = stock.quote.history(
                start=(dt.date.today() - dt.timedelta(days=10)).strftime("%Y-%m-%d"),
                end=dt.date.today().strftime("%Y-%m-%d"),
            )
            if history is not None and not history.empty:
                latest = history.iloc[-1]
                return {
                    "price": float(latest["close"]),
                    "timestamp": self._normalize_timestamp(latest.get("time")),
                }
        except Exception as exc:
            logger.warning("paper_trade_price_fetch_failed", ticker=ticker, error=str(exc))

        fallback_df = self._load_ohlcv(ticker, use_mock=True)
        return {
            "price": float(fallback_df["close"].iloc[-1]),
            "timestamp": dt.datetime.now(dt.UTC),
        }

    def _fetch_news_context(self, ticker: str) -> str:
        """Fetch the latest company news headlines for the LLM prompt."""
        try:
            import os
            from vnstock import Vnstock

            settings = get_settings()
            os.environ["VNAI_API_KEY"] = settings.vnstock_api_key
            os.environ["VNSTOCK_API_KEY"] = settings.vnstock_api_key

            stock = Vnstock().stock(symbol=ticker)
            news_df = stock.company.news()
            if news_df is None or news_df.empty:
                return ""

            headlines = []
            for _, row in news_df.head(5).iterrows():
                headlines.append(f"- {row['title']} ({row['publish_time']})")
            return "\n".join(headlines)
        except Exception as exc:
            logger.warning("paper_trade_news_fetch_failed", ticker=ticker, error=str(exc))
            return ""

    def _apply_execution(
        self,
        ticker: str,
        decision_action: str,
        order_payload: dict[str, Any] | None,
        current_price: float,
        as_of: dt.datetime,
        latency: LatencyProfile,
    ) -> dict[str, Any]:
        """Apply paper fills and update cash and position state."""
        open_position = self._open_positions.get(ticker)

        if decision_action == "EXECUTE_BUY" and order_payload:
            if open_position is not None:
                return {
                    "status": "already_in_position",
                    "trade_id": open_position.trade_id,
                    "volume": open_position.volume,
                }

            volume = int(order_payload.get("volume", 0))
            if volume <= 0:
                return {"status": "blocked_zero_volume"}

            requested_price = float(order_payload.get("entry_price", current_price))
            filled_price, fees, slippage_cost = simulate_execution_cost(
                entry_price=requested_price,
                volume=volume,
                action="BUY",
            )
            total_cost = filled_price * volume + fees
            if total_cost > self._cash:
                return {
                    "status": "insufficient_cash",
                    "required_cash": round(total_cost, 2),
                    "available_cash": round(self._cash, 2),
                }

            trade = PaperTrade(
                trade_id=str(uuid.uuid4())[:8],
                ticker=ticker,
                action="BUY",
                entry_price=round(filled_price, 2),
                volume=volume,
                fees=round(fees, 2),
                slippage=round(slippage_cost, 2),
                entry_time=as_of.isoformat(),
                last_mark_price=current_price,
                latency=latency,
                decision=decision_action,
                status="OPEN",
            )
            self._cash -= total_cost
            self._open_positions[ticker] = trade
            self._trades.append(trade)
            return {
                "status": "opened",
                "trade_id": trade.trade_id,
                "filled_price": round(filled_price, 2),
                "volume": volume,
                "cash_spent": round(total_cost, 2),
            }

        if decision_action == "EXECUTE_SELL":
            if open_position is None:
                return {"status": "no_position_to_sell"}

            volume = open_position.volume
            filled_price, fees, slippage_cost = simulate_execution_cost(
                entry_price=current_price,
                volume=volume,
                action="SELL",
            )
            entry_cost = open_position.entry_price * volume + open_position.fees
            net_proceeds = filled_price * volume - fees
            realized_pnl = net_proceeds - entry_cost

            open_position.action = "SELL"
            open_position.exit_price = round(filled_price, 2)
            open_position.exit_time = as_of.isoformat()
            open_position.fees = round(open_position.fees + fees, 2)
            open_position.slippage = round(open_position.slippage + slippage_cost, 2)
            open_position.last_mark_price = current_price
            open_position.pnl = round(realized_pnl, 2)
            open_position.pnl_pct = round((realized_pnl / entry_cost) * 100, 2) if entry_cost else 0.0
            open_position.decision = decision_action
            open_position.status = "CLOSED"

            self._cash += net_proceeds
            self._realized_pnl += realized_pnl
            del self._open_positions[ticker]
            return {
                "status": "closed",
                "trade_id": open_position.trade_id,
                "filled_price": round(filled_price, 2),
                "volume": volume,
                "realized_pnl": round(realized_pnl, 2),
            }

        return {"status": "no_action"}

    def _mark_to_market(self, ticker: str, current_price: float) -> None:
        """Update the latest observable price for open positions."""
        if ticker in self._open_positions:
            self._open_positions[ticker].last_mark_price = current_price

    @staticmethod
    def _normalize_timestamp(value: Any) -> dt.datetime:
        """Normalize timestamp-like values into aware UTC datetimes."""
        if isinstance(value, dt.datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=dt.UTC)
            return value.astimezone(dt.UTC)
        if isinstance(value, dt.date):
            return dt.datetime.combine(value, dt.time.min, tzinfo=dt.UTC)
        if value:
            raw = str(value)
            try:
                parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=dt.UTC)
                return parsed.astimezone(dt.UTC)
            except ValueError:
                pass
        return dt.datetime.now(dt.UTC)


def validate_latency(profile: LatencyProfile) -> bool:
    """Return False when the total latency breaches the configured SLA."""
    threshold = get_settings().latency_sla_seconds
    if profile.total_latency_seconds > threshold:
        logger.error(
            "latency_breach",
            total=profile.total_latency_seconds,
            llm=profile.llm_inference_ms,
            threshold=threshold,
        )
        return False
    return True


def track_execution_slippage(
    ideal_json_price: float,
    actual_filled_price: float,
    action: str = "BUY",
) -> float:
    """Calculate execution slippage as a percentage of the intended price."""
    if action == "BUY":
        slippage_pct = (ideal_json_price - actual_filled_price) / ideal_json_price
    else:
        slippage_pct = (actual_filled_price - ideal_json_price) / ideal_json_price
    return slippage_pct


async def profile_latency_wrapper(func, *args, **kwargs) -> tuple[float, Any]:
    """Wrap async tasks to measure latency in milliseconds."""
    started_at = time.perf_counter()
    result = await func(*args, **kwargs)
    return (time.perf_counter() - started_at) * 1000, result
