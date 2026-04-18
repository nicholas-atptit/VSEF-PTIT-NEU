"""Phase 1 and Phase 2 reporting helpers."""

from .manifests import (
    build_run_manifest,
    collect_dependency_versions,
    collect_git_metadata,
    collect_runtime_metadata,
    write_run_manifest,
)
from .summary import (
    build_conditioning_mode_summary,
    build_model_comparison_summary,
    build_phase2_conditioning_summary,
    render_phase2_summary_markdown,
    render_summary_markdown,
    write_summary_markdown,
    write_summary_tables,
)

__all__ = [
    "collect_git_metadata",
    "collect_runtime_metadata",
    "collect_dependency_versions",
    "build_run_manifest",
    "write_run_manifest",
    "build_conditioning_mode_summary",
    "build_model_comparison_summary",
    "build_phase2_conditioning_summary",
    "write_summary_tables",
    "render_phase2_summary_markdown",
    "render_summary_markdown",
    "write_summary_markdown",
]
