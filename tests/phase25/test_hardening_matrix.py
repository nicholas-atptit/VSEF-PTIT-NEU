from __future__ import annotations

from src.evaluation.hardening import (
    build_phase25_core_frame,
    build_phase25_matrix_config,
    build_phase25_sweep_frame,
)


def test_medium_matrix_expands_to_expected_core_and_policy_runs() -> None:
    matrix = build_phase25_matrix_config("medium")

    core_frame = build_phase25_core_frame(matrix)
    sweep_frame = build_phase25_sweep_frame(matrix)

    assert len(core_frame) == 9
    assert set(core_frame["group_name"]) == {"small_banks", "mixed_large_cap", "vn100_subset"}
    assert set(core_frame["horizon"]) == {1, 5, 10}

    assert len(sweep_frame) == 72
    assert set(sweep_frame["cost_mode"]) == {"baseline", "elevated"}
    assert set(sweep_frame["sizing_mode"]) == {"adaptive", "fixed_fraction"}
    assert set(sweep_frame["threshold"]) == {0.003, 0.007}
