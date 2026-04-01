"""FastAPI application entrypoint for the trading system."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator
import sys
import os

print(f"DEBUG: ROOT_FILE={__file__}")
print(f"DEBUG: CWD={os.getcwd()}")
print(f"DEBUG: SYS_PATH_0={sys.path[0]}")

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pathlib import Path

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
    title="Algo Trading System - Full Pipeline (Phase 1-5)",
    description=(
        "End-to-end algorithmic trading system integrating: "
        "real-time data infrastructure, quantitative ML, qualitative LLM analysis, "
        "decision matrix, risk management, and paper trading."
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

# Mount web dashboard static files
_web_dir = Path(__file__).resolve().parent.parent.parent / "web"
if _web_dir.exists():
    app.mount("/web", StaticFiles(directory=str(_web_dir), html=True), name="web-dashboard")


@app.get("/", tags=["Root"])
async def root() -> dict:
    """Return a compact service overview."""
    return {
        "service": "Algo Trading System - Phase 1-5",
        "version": "5.0.0",
        "docs": "/docs",
        "predict": "/api/v1/predict?ticker=SSI",
        "endpoints": {
            "health": "/api/v1/health",
            "predict": "/api/v1/predict?ticker=SSI",
            "analyze": "/api/v1/analyze?ticker=SSI",
            "execute": "/api/v1/execute?ticker=SSI",
            "paper_trade": "/api/v1/paper-trade?ticker=SSI",
            "ingest_news": "POST /api/v1/ingest-news",
            "ingest_bctc": "POST /api/v1/ingest-bctc",
        },
    }


@app.get("/dashboard", include_in_schema=False)
async def dashboard_redirect():
    """Redirect to the web dashboard."""
    return RedirectResponse(url="/web/index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Silence browser favicon requests."""
    return Response(status_code=204)
