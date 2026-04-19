"""Role-aware model governance helpers for quant-core orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import pandas as pd


ModelRole = Literal[
    "primary_research",
    "comparator",
    "baseline_only",
    "parked",
    "shadow_only",
]

RunMode = Literal[
    "full_forecast",
    "research_core",
    "decision_core",
    "baseline_only",
]

MODEL_ROLES: tuple[ModelRole, ...] = (
    "primary_research",
    "comparator",
    "baseline_only",
    "parked",
    "shadow_only",
)

RUN_MODES: tuple[RunMode, ...] = (
    "full_forecast",
    "research_core",
    "decision_core",
    "baseline_only",
)

MODEL_STATUSES: tuple[str, ...] = (
    "active",
    "baseline",
    "shadow",
    "parked",
)


@dataclass(frozen=True)
class RunModeSpec:
    """Canonical run-mode metadata used by the quant core."""

    run_mode: RunMode
    description: str
    enablement_field: str
    included_roles: tuple[ModelRole, ...]
    analysis_first: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "run_mode": self.run_mode,
            "description": self.description,
            "enablement_field": self.enablement_field,
            "included_roles": list(self.included_roles),
            "analysis_first": bool(self.analysis_first),
        }


RUN_MODE_SPECS: dict[RunMode, RunModeSpec] = {
    "full_forecast": RunModeSpec(
        run_mode="full_forecast",
        description="Run the full supported quant-core forecast zoo and collect full analysis outputs.",
        enablement_field="enabled_for_full_forecast",
        included_roles=("primary_research", "comparator", "baseline_only", "shadow_only"),
        analysis_first=True,
    ),
    "research_core": RunModeSpec(
        run_mode="research_core",
        description="Run the current research lane using primary and comparator models only.",
        enablement_field="enabled_for_research_core",
        included_roles=("primary_research", "comparator"),
        analysis_first=True,
    ),
    "decision_core": RunModeSpec(
        run_mode="decision_core",
        description="Run only the decision-authorized model subset for downstream policy candidate generation.",
        enablement_field="enabled_for_decision_core",
        included_roles=("primary_research", "comparator"),
        analysis_first=False,
    ),
    "baseline_only": RunModeSpec(
        run_mode="baseline_only",
        description="Run baseline statistical checkpoints only.",
        enablement_field="baseline_only",
        included_roles=("baseline_only",),
        analysis_first=True,
    ),
}


@dataclass(frozen=True)
class ModelGovernanceEntry:
    """Serializable model-governance metadata used across quant-core layers."""

    model_name: str
    family: str
    status: str
    research_priority: int
    role: ModelRole
    enabled_for_full_forecast: bool
    enabled_for_research_core: bool
    enabled_for_decision_core: bool
    baseline_only: bool
    comparator_only: bool
    parked: bool
    supports_return: bool
    supports_direction: bool
    supports_policy_eval: bool
    notes: str = ""

    def __post_init__(self) -> None:
        if self.role not in MODEL_ROLES:
            raise ValueError(f"Unsupported model role '{self.role}'. Expected one of {MODEL_ROLES}.")
        if self.status not in MODEL_STATUSES:
            raise ValueError(f"Unsupported model status '{self.status}'. Expected one of {MODEL_STATUSES}.")
        if self.parked and self.role != "parked":
            raise ValueError("parked entries must use the 'parked' role")
        if self.role == "baseline_only" and not self.baseline_only:
            raise ValueError("baseline_only role entries must set baseline_only=True")

    def is_enabled_for_run_mode(self, run_mode: str) -> bool:
        spec = get_run_mode_spec(run_mode)
        if self.role not in spec.included_roles:
            return False
        return bool(getattr(self, spec.enablement_field))

    def sort_key(self) -> tuple[int, str, str]:
        return (int(self.research_priority), str(self.family), str(self.model_name))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "family": self.family,
            "status": self.status,
            "research_priority": int(self.research_priority),
            "role": self.role,
            "enabled_for_full_forecast": bool(self.enabled_for_full_forecast),
            "enabled_for_research_core": bool(self.enabled_for_research_core),
            "enabled_for_decision_core": bool(self.enabled_for_decision_core),
            "baseline_only": bool(self.baseline_only),
            "comparator_only": bool(self.comparator_only),
            "parked": bool(self.parked),
            "supports_return": bool(self.supports_return),
            "supports_direction": bool(self.supports_direction),
            "supports_policy_eval": bool(self.supports_policy_eval),
            "notes": self.notes,
        }


def normalize_model_role(role: str) -> ModelRole:
    key = str(role).strip().lower()
    if key not in MODEL_ROLES:
        raise ValueError(f"Unsupported model role '{role}'. Available: {MODEL_ROLES}")
    return key  # type: ignore[return-value]


def normalize_run_mode(run_mode: str | None) -> RunMode:
    key = str(run_mode or "full_forecast").strip().lower()
    if key not in RUN_MODES:
        raise ValueError(f"Unsupported run mode '{run_mode}'. Available: {RUN_MODES}")
    return key  # type: ignore[return-value]


def get_run_mode_spec(run_mode: str | None) -> RunModeSpec:
    return RUN_MODE_SPECS[normalize_run_mode(run_mode)]


def run_mode_table() -> pd.DataFrame:
    rows = [spec.to_dict() for spec in RUN_MODE_SPECS.values()]
    return pd.DataFrame(rows).sort_values("run_mode").reset_index(drop=True)


def infer_target_support(target_type: str | None) -> str | None:
    if target_type is None:
        return None
    key = str(target_type).strip().lower()
    if "direction" in key:
        return "direction"
    return "return"


def filter_governance_entries(
    entries: Iterable[ModelGovernanceEntry],
    *,
    run_mode: str | None = "full_forecast",
    roles: Iterable[str] | None = None,
    model_names: Iterable[str] | None = None,
    target_type: str | None = None,
    require_policy_eval: bool = False,
    include_parked: bool = False,
) -> list[ModelGovernanceEntry]:
    allowed_roles = {normalize_model_role(role) for role in roles} if roles else None
    allowed_names = {str(name).strip().lower() for name in model_names} if model_names else None
    target_support = infer_target_support(target_type)
    resolved_run_mode = normalize_run_mode(run_mode)

    selected: list[ModelGovernanceEntry] = []
    for entry in entries:
        if allowed_names is not None and entry.model_name.lower() not in allowed_names:
            continue
        if allowed_roles is not None and entry.role not in allowed_roles:
            continue
        if not include_parked and entry.parked:
            continue
        if not entry.is_enabled_for_run_mode(resolved_run_mode):
            continue
        if target_support == "return" and not entry.supports_return:
            continue
        if target_support == "direction" and not entry.supports_direction:
            continue
        if require_policy_eval and not entry.supports_policy_eval:
            continue
        selected.append(entry)
    return sorted(selected, key=lambda entry: entry.sort_key())


def governance_table(
    entries: Iterable[ModelGovernanceEntry],
    *,
    run_mode: str | None = None,
    roles: Iterable[str] | None = None,
    model_names: Iterable[str] | None = None,
    target_type: str | None = None,
    require_policy_eval: bool = False,
    include_parked: bool = True,
) -> pd.DataFrame:
    selected = (
        filter_governance_entries(
            entries,
            run_mode=run_mode,
            roles=roles,
            model_names=model_names,
            target_type=target_type,
            require_policy_eval=require_policy_eval,
            include_parked=include_parked,
        )
        if any(value is not None for value in (run_mode, roles, model_names, target_type)) or require_policy_eval
        else sorted(list(entries), key=lambda entry: entry.sort_key())
    )
    if not selected:
        return pd.DataFrame(columns=list(ModelGovernanceEntry("x", "x", "active", 0, "comparator", True, False, False, False, False, False, True, True, True).to_dict()))
    return pd.DataFrame([entry.to_dict() for entry in selected]).sort_values(
        ["research_priority", "family", "model_name"]
    ).reset_index(drop=True)
