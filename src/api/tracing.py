"""Shared request tracing utilities for FastAPI endpoints."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RequestTrace:
    """Collect per-stage timing for a single request."""

    path: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: float = field(default_factory=time.perf_counter)
    stages_ms: dict[str, float] = field(default_factory=dict)

    def record_stage(self, stage_name: str, elapsed_ms: float) -> None:
        """Accumulate elapsed time for a stage."""
        self.stages_ms[stage_name] = round(self.stages_ms.get(stage_name, 0.0) + elapsed_ms, 1)

    def as_header_value(self) -> str:
        """Serialize stages to a compact response header."""
        return ";".join(f"{name}={value:.1f}" for name, value in self.stages_ms.items())


def get_request_trace(request: Request | None) -> RequestTrace | None:
    """Read the request trace object from FastAPI state."""
    if request is None:
        return None
    return getattr(request.state, "request_trace", None)


@contextmanager
def trace_stage(request: Request | None, stage_name: str):
    """Measure an endpoint stage and store the elapsed time on the request."""
    started_at = time.perf_counter()
    try:
        yield
    finally:
        trace = get_request_trace(request)
        if trace is not None:
            trace.record_stage(stage_name, (time.perf_counter() - started_at) * 1000)


class LatencyTracingMiddleware(BaseHTTPMiddleware):
    """Attach total and per-stage latency metrics to each API response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        settings = get_settings()
        trace = RequestTrace(path=request.url.path)
        request.state.request_trace = trace

        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - trace.started_at) * 1000
        response.headers["X-Trace-Id"] = trace.trace_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        if trace.stages_ms:
            response.headers["X-Stage-Timings"] = trace.as_header_value()

        if request.url.path.startswith("/api/"):
            logger.info(
                "request_latency",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                trace_id=trace.trace_id,
                latency_ms=round(elapsed_ms, 1),
                stages=trace.stages_ms,
            )
            if elapsed_ms > settings.latency_sla_seconds * 1000:
                logger.warning(
                    "sla_breach",
                    path=request.url.path,
                    trace_id=trace.trace_id,
                    latency_ms=round(elapsed_ms, 1),
                    threshold_ms=settings.latency_sla_seconds * 1000,
                )

        return response
