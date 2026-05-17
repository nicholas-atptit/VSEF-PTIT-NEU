"""FastAPI application entrypoint for the diagnostic research API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings
from src.api.routes import router as v1_router
from src.api.routes_v2 import router as v2_router
from src.api.tracing import LatencyTracingMiddleware
from src.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle."""
    settings = get_settings()
    logger.info(
        "api_startup",
        host=settings.api_host,
        port=settings.api_port,
        model_dir=settings.model_dir,
    )
    yield
    logger.info("api_shutdown")


app = FastAPI(
    title="Diagnostic Research API",
    description=(
        "Forecast, risk, scenario, and route diagnostics for research workflows. "
        "The service is non-executing, non-advisory, and not connected to account routing."
    ),
    version="5.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LatencyTracingMiddleware)
app.include_router(v1_router)
app.include_router(v2_router)


@app.get("/", tags=["Root"])
async def root() -> dict:
    """Return a compact diagnostic service overview."""
    return {
        "service": "Diagnostic Research API",
        "version": "5.0.0",
        "docs": "/docs",
        "scope": [
            "forecast_diagnostics",
            "risk_diagnostics",
            "scenario_diagnostics",
            "route_diagnostics",
        ],
        "boundaries": [
            "research_only",
            "non_executing",
            "non_advisory",
            "no_account_routing",
        ],
        "predict": "/api/v1/predict?ticker=SSI",
        "endpoints": {
            "health": "/api/v1/health",
            "predict": "/api/v1/predict?ticker=SSI",
            "analyze": "/api/v1/analyze?ticker=SSI",
            "chat": "POST /api/v1/chat",
            "market_depth": "/api/v1/order-book?ticker=SSI",
        },
    }


@app.get("/dashboard", include_in_schema=False)
async def dashboard_removed() -> dict:
    """Report that the web UI is no longer served by the governed runtime."""
    return {
        "status": "removed",
        "detail": "The web dashboard is no longer part of the governed runtime.",
        "docs": "/docs",
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Silence browser favicon requests."""
    return Response(status_code=204)
