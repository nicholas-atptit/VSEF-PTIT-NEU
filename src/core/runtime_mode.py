"""Runtime mode and provenance helpers for governed execution paths."""

from __future__ import annotations

from enum import Enum
from typing import Any


class RuntimeMode(str, Enum):
    """Explicit execution modes used by governed runtime surfaces."""

    DEMO = "demo"
    RESEARCH = "research"
    AUDIT = "audit"


RUNTIME_MODE_VALUES = tuple(mode.value for mode in RuntimeMode)
MOCK_SOURCE = "synthetic_mock_data"


def normalize_runtime_mode(mode: str | RuntimeMode | None = None) -> RuntimeMode:
    """Return a canonical runtime mode, defaulting to research."""

    if isinstance(mode, RuntimeMode):
        return mode
    if mode is None:
        return RuntimeMode.RESEARCH
    normalized = str(mode).strip().lower()
    try:
        return RuntimeMode(normalized)
    except ValueError as exc:
        allowed = ", ".join(RUNTIME_MODE_VALUES)
        raise ValueError(f"Invalid runtime_mode '{mode}'. Expected one of: {allowed}") from exc


def ensure_mock_allowed(
    runtime_mode: str | RuntimeMode | None,
    *,
    explicit_mock: bool = False,
    fallback_triggered: bool = False,
) -> RuntimeMode:
    """Raise when mock data would violate the active runtime mode."""

    mode = normalize_runtime_mode(runtime_mode)
    if mode is RuntimeMode.AUDIT and (explicit_mock or fallback_triggered):
        raise RuntimeError("Mock data disabled for audit mode")
    if mode is RuntimeMode.RESEARCH and fallback_triggered:
        raise RuntimeError("Mock fallback disabled for research mode")
    return mode


def build_data_provenance(
    *,
    source: str,
    uses_mock_data: bool,
    fallback_triggered: bool,
    runtime_mode: str | RuntimeMode | None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Build the additive data provenance payload used by runtime outputs."""

    payload: dict[str, Any] = {
        "source": source,
        "uses_mock_data": bool(uses_mock_data),
        "fallback_triggered": bool(fallback_triggered),
        "runtime_mode": normalize_runtime_mode(runtime_mode).value,
    }
    if reason:
        payload["reason"] = reason
    return payload

