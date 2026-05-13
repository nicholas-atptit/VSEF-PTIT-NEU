"""Audit local hourly VN30 data and select an available-window research design."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_hourly_common import (  # noqa: E402
    VN30_TICKERS,
    load_hourly_sources_for_ticker,
    read_universe,
)
from scripts.research.vn30_hourly_available_window_common import (  # noqa: E402
    AUDIT_DIR,
    DESIGN_DECISION_JSON,
    DESIGN_DECISION_MD,
    MIN_COMMON_TIMESTAMPS,
    MIN_EVAL_ROWS_PER_TICKER,
    MIN_TRAIN_ROWS_PER_TICKER,
    REPORT_ROOT,
    TRAIN_FRACTION,
    markdown_table,
    rel,
    timestamp_text,
    write_csv,
    write_docx_build_notes,
    write_json,
)


DOCX_NOTES_PATH = REPO_ROOT / "reports" / "NCKH_VN30_HOURLY_AVAILABLE_WINDOW_DOCX_BUILD_NOTES.md"
AUDIT_COLUMNS = [
    "ticker",
    "first_available_hourly_timestamp",
    "last_available_hourly_timestamp",
    "hourly_rows",
    "rows_before_2025",
    "rows_2025_onward",
    "raw_hourly_sources",
    "raw_hourly_files",
]
WINDOW_COLUMNS = [
    "coverage_floor",
    "ticker_count",
    "window_start",
    "window_end",
    "common_timestamp_count",
    "min_rows_per_ticker",
    "train_rows_per_ticker",
    "eval_rows_per_ticker",
    "training_start",
    "training_cutoff",
    "evaluation_start",
    "evaluation_end",
    "valid_split",
    "selected_tickers",
]
INDEX_CODES = ["VNINDEX", "VN30INDEX", "VNXALL"]
INDEX_AUDIT_COLUMNS = [
    "index_code",
    "exact_code_local_file_found",
    "first_available_hourly_timestamp",
    "last_available_hourly_timestamp",
    "hourly_rows",
    "selected_window_start",
    "selected_window_end",
    "selected_window_rows",
    "selected_train_rows",
    "selected_eval_rows",
    "overlaps_selected_window",
    "covers_selected_window",
    "available_window_context_ready",
    "limitation",
    "raw_hourly_sources",
    "raw_hourly_files",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local VN30 hourly data for an available-window research package.")
    parser.add_argument("--output-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--decision-md", type=Path, default=DESIGN_DECISION_MD)
    parser.add_argument("--decision-json", type=Path, default=DESIGN_DECISION_JSON)
    return parser.parse_args()


def ticker_audit_rows() -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    for ticker in read_universe():
        frame, sources, files = load_hourly_sources_for_ticker(ticker)
        frames[ticker] = frame
        timestamps = pd.to_datetime(frame["datetime"], errors="coerce").dropna() if not frame.empty else pd.Series(dtype="datetime64[ns]")
        first_ts = pd.Timestamp(timestamps.min()) if not timestamps.empty else None
        last_ts = pd.Timestamp(timestamps.max()) if not timestamps.empty else None
        rows.append(
            {
                "ticker": ticker,
                "first_available_hourly_timestamp": timestamp_text(first_ts),
                "last_available_hourly_timestamp": timestamp_text(last_ts),
                "hourly_rows": int(len(frame)),
                "rows_before_2025": int((timestamps < pd.Timestamp("2025-01-01 00:00:00")).sum()) if not timestamps.empty else 0,
                "rows_2025_onward": int((timestamps >= pd.Timestamp("2025-01-01 00:00:00")).sum()) if not timestamps.empty else 0,
                "raw_hourly_sources": ";".join(sources),
                "raw_hourly_files": ";".join(files),
            }
        )
    return rows, frames


def first_last_maps(rows: list[dict[str, Any]]) -> tuple[dict[str, pd.Timestamp], dict[str, pd.Timestamp]]:
    first: dict[str, pd.Timestamp] = {}
    last: dict[str, pd.Timestamp] = {}
    for row in rows:
        ticker = str(row["ticker"])
        first_text = str(row.get("first_available_hourly_timestamp", ""))
        last_text = str(row.get("last_available_hourly_timestamp", ""))
        if first_text and last_text:
            first[ticker] = pd.Timestamp(first_text)
            last[ticker] = pd.Timestamp(last_text)
    return first, last


def timestamp_intersection(
    frames: dict[str, pd.DataFrame],
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[pd.Timestamp]:
    common: set[pd.Timestamp] | None = None
    for ticker in tickers:
        frame = frames.get(ticker, pd.DataFrame())
        if frame.empty:
            return []
        values = set(
            pd.Timestamp(value)
            for value in frame.loc[(frame["datetime"] >= start) & (frame["datetime"] <= end), "datetime"].dropna()
        )
        common = values if common is None else common.intersection(values)
        if not common:
            return []
    return sorted(common or set())


def ticker_rows_between(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> int:
    if frame.empty:
        return 0
    return int(((frame["datetime"] >= start) & (frame["datetime"] <= end)).sum())


def split_from_common_timestamps(common_timestamps: list[pd.Timestamp]) -> dict[str, Any]:
    total = len(common_timestamps)
    if total < MIN_COMMON_TIMESTAMPS:
        return {
            "valid_split": False,
            "train_rows_per_ticker": 0,
            "eval_rows_per_ticker": 0,
            "training_start": "",
            "training_cutoff": "",
            "evaluation_start": "",
            "evaluation_end": "",
            "reason": f"common_timestamps_below_min:{total}<{MIN_COMMON_TIMESTAMPS}",
        }
    split_idx = max(MIN_TRAIN_ROWS_PER_TICKER, int(total * TRAIN_FRACTION))
    split_idx = min(split_idx, total - MIN_EVAL_ROWS_PER_TICKER)
    train_rows = split_idx
    eval_rows = total - split_idx
    if train_rows < MIN_TRAIN_ROWS_PER_TICKER or eval_rows < MIN_EVAL_ROWS_PER_TICKER:
        return {
            "valid_split": False,
            "train_rows_per_ticker": int(train_rows),
            "eval_rows_per_ticker": int(eval_rows),
            "training_start": timestamp_text(common_timestamps[0]) if common_timestamps else "",
            "training_cutoff": "",
            "evaluation_start": "",
            "evaluation_end": timestamp_text(common_timestamps[-1]) if common_timestamps else "",
            "reason": "train_or_eval_rows_below_min",
        }
    return {
        "valid_split": True,
        "train_rows_per_ticker": int(train_rows),
        "eval_rows_per_ticker": int(eval_rows),
        "training_start": timestamp_text(common_timestamps[0]),
        "training_cutoff": timestamp_text(common_timestamps[split_idx - 1]),
        "evaluation_start": timestamp_text(common_timestamps[split_idx]),
        "evaluation_end": timestamp_text(common_timestamps[-1]),
        "reason": "valid_common_timestamp_split",
    }


def build_candidate(
    frames: dict[str, pd.DataFrame],
    first: dict[str, pd.Timestamp],
    last: dict[str, pd.Timestamp],
    start: pd.Timestamp,
    coverage_floor: int,
) -> dict[str, Any]:
    tickers = sorted([ticker for ticker, first_ts in first.items() if first_ts <= start and last.get(ticker, pd.Timestamp.min) >= start])
    if not tickers:
        end = start
    else:
        end = min(last[ticker] for ticker in tickers)
    common_timestamps = timestamp_intersection(frames, tickers, start, end) if tickers else []
    row_counts = [ticker_rows_between(frames[ticker], start, end) for ticker in tickers]
    split = split_from_common_timestamps(common_timestamps)
    return {
        "coverage_floor": coverage_floor,
        "ticker_count": int(len(tickers)),
        "window_start": timestamp_text(start),
        "window_end": timestamp_text(end),
        "common_timestamp_count": int(len(common_timestamps)),
        "min_rows_per_ticker": min(row_counts) if row_counts else 0,
        "train_rows_per_ticker": split["train_rows_per_ticker"],
        "eval_rows_per_ticker": split["eval_rows_per_ticker"],
        "training_start": split["training_start"],
        "training_cutoff": split["training_cutoff"],
        "evaluation_start": split["evaluation_start"],
        "evaluation_end": split["evaluation_end"],
        "valid_split": bool(split["valid_split"]),
        "split_reason": split["reason"],
        "selected_tickers": ",".join(tickers),
        "ticker_list": tickers,
    }


def coverage_windows(rows: list[dict[str, Any]], frames: dict[str, pd.DataFrame]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    first, last = first_last_maps(rows)
    unique_starts = sorted(set(first.values()))
    candidates = [build_candidate(frames, first, last, start, coverage_floor=0) for start in unique_starts]
    threshold_rows: list[dict[str, Any]] = []
    for threshold in (30, 25, 20, 15, 10):
        eligible = [candidate for candidate in candidates if int(candidate["ticker_count"]) >= threshold]
        if not eligible:
            threshold_rows.append(
                {
                    "coverage_floor": threshold,
                    "ticker_count": 0,
                    "window_start": "",
                    "window_end": "",
                    "common_timestamp_count": 0,
                    "min_rows_per_ticker": 0,
                    "train_rows_per_ticker": 0,
                    "eval_rows_per_ticker": 0,
                    "training_start": "",
                    "training_cutoff": "",
                    "evaluation_start": "",
                    "evaluation_end": "",
                    "valid_split": False,
                    "selected_tickers": "",
                }
            )
            continue
        earliest = min(eligible, key=lambda item: pd.Timestamp(item["window_start"]))
        threshold_rows.append({**earliest, "coverage_floor": threshold})
    feasible_candidates = [candidate for candidate in candidates if candidate["ticker_count"] >= 20 and candidate["valid_split"]]
    return threshold_rows, feasible_candidates


def select_design(
    rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    feasible_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    full_common = next((row for row in threshold_rows if int(row["coverage_floor"]) == 30), None)
    selected: dict[str, Any] | None = None
    selected_priority = 0
    for priority in (30, 25, 20):
        eligible = [candidate for candidate in feasible_candidates if int(candidate["ticker_count"]) >= priority]
        if eligible:
            selected = sorted(
                eligible,
                key=lambda item: (
                    int(item["ticker_count"]),
                    int(item["common_timestamp_count"]),
                    -pd.Timestamp(item["window_start"]).value,
                ),
                reverse=True,
            )[0]
            selected_priority = priority
            break

    if selected is None:
        return {
            "final_paper_can_proceed": False,
            "selected_priority": 0,
            "selected_tickers": [],
            "excluded_tickers": VN30_TICKERS,
            "exclusion_reasons": {ticker: "no_valid_20_ticker_or_larger_common_hourly_train_eval_split" for ticker in VN30_TICKERS},
            "training_start": "",
            "training_cutoff": "",
            "evaluation_start": "",
            "evaluation_end": "",
            "frequency": "hourly",
            "minimum_row_count": 0,
            "full_vn30_representativeness": False,
            "claim_boundary": "No available-window final paper: fewer than 20 tickers have a valid common hourly split.",
            "full_30_common_window_feasible": bool(full_common and full_common.get("valid_split")),
            "full_30_common_start": full_common.get("window_start", "") if full_common else "",
            "full_30_common_end": full_common.get("window_end", "") if full_common else "",
        }

    chosen = [ticker for ticker in selected["ticker_list"]]
    excluded = [ticker for ticker in VN30_TICKERS if ticker not in set(chosen)]
    first_lookup = {row["ticker"]: row.get("first_available_hourly_timestamp", "") for row in rows}
    exclusion_reasons = {
        ticker: f"first_available_after_selected_window_start:{first_lookup.get(ticker, '')}>{selected['window_start']}"
        for ticker in excluded
    }
    subset_phrase = (
        "Full 30/30 frozen VN30 available-window analysis."
        if len(chosen) == 30
        else "The study is an hourly available-window VN30 subset analysis rather than a full-constituent VN30 historical evaluation."
    )
    return {
        "final_paper_can_proceed": len(chosen) >= 20,
        "selected_priority": selected_priority,
        "selected_tickers": chosen,
        "excluded_tickers": excluded,
        "exclusion_reasons": exclusion_reasons,
        "training_start": selected["training_start"],
        "training_cutoff": selected["training_cutoff"],
        "evaluation_start": selected["evaluation_start"],
        "evaluation_end": selected["evaluation_end"],
        "frequency": "hourly",
        "minimum_row_count": int(selected["common_timestamp_count"]),
        "minimum_train_rows_per_ticker": int(selected["train_rows_per_ticker"]),
        "minimum_eval_rows_per_ticker": int(selected["eval_rows_per_ticker"]),
        "selected_common_window_start": selected["window_start"],
        "selected_common_window_end": selected["window_end"],
        "selected_ticker_count": len(chosen),
        "full_vn30_representativeness": len(chosen) == 30,
        "claim_boundary": subset_phrase,
        "full_30_common_window_feasible": bool(full_common and full_common.get("valid_split")),
        "full_30_common_start": full_common.get("window_start", "") if full_common else "",
        "full_30_common_end": full_common.get("window_end", "") if full_common else "",
        "full_30_common_timestamp_count": int(full_common.get("common_timestamp_count", 0)) if full_common else 0,
        "daily_data_used": False,
        "daily_to_hourly_resampling_used": False,
        "vn100_evidence_reused": False,
        "selection_rule": "priority: 30 valid, then >=25 valid, then >=20 valid; within priority maximize ticker count then common timestamps",
        "minimum_split_rule": (
            f"common timestamps >= {MIN_COMMON_TIMESTAMPS}, train rows >= {MIN_TRAIN_ROWS_PER_TICKER}, "
            f"eval rows >= {MIN_EVAL_ROWS_PER_TICKER}"
        ),
    }


def write_audit_report(
    path: Path,
    rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    design: dict[str, Any],
) -> None:
    full_feasible = bool(design.get("full_30_common_window_feasible"))
    best_text = (
        f"{design.get('selected_ticker_count', 0)} tickers from {design.get('training_start', '')} "
        f"to {design.get('evaluation_end', '')}"
        if design.get("final_paper_can_proceed")
        else "No >=20-ticker valid common hourly train/eval split."
    )
    content = [
        "# VN30 Hourly Available-Window Audit",
        "",
        "## Scope",
        "",
        "- Universe: frozen VN30, exactly 30 tickers.",
        "- Frequency: hourly only.",
        "- Source: local hourly files only.",
        "- Daily data, daily-to-hourly resampling, and old VN100 artifacts are not used.",
        "",
        "## Direct Answers",
        "",
        f"- Is a full-30 common hourly window feasible? {str(full_feasible).lower()}.",
        (
            f"- Full-30 common window: {design.get('full_30_common_start', '')} to {design.get('full_30_common_end', '')}; "
            f"common timestamps: {design.get('full_30_common_timestamp_count', 0)}."
            if design.get("full_30_common_start")
            else "- Full-30 common window: unavailable."
        ),
        (
            "- If no: the full-30 common window does not meet the minimum train/eval row rule for this available-window study."
            if not full_feasible
            else "- If yes: the full-30 common window meets the minimum train/eval row rule."
        ),
        f"- Best available-window design: {best_text}.",
        f"- Final paper can proceed: {str(bool(design.get('final_paper_can_proceed'))).lower()}.",
        "",
        "## Coverage Windows",
        "",
        markdown_table(
            [
                "coverage_floor",
                "ticker_count",
                "window_start",
                "window_end",
                "common_timestamp_count",
                "min_rows_per_ticker",
                "train_rows_per_ticker",
                "eval_rows_per_ticker",
                "valid_split",
            ],
            threshold_rows,
        ),
        "",
        "## Per-Ticker Local Hourly Coverage",
        "",
        markdown_table(
            [
                "ticker",
                "first_available_hourly_timestamp",
                "last_available_hourly_timestamp",
                "hourly_rows",
                "rows_before_2025",
                "rows_2025_onward",
            ],
            rows,
        ),
        "",
        "## Limitations",
        "",
        "- This audit uses only the real local hourly data already present in the repository.",
        "- It does not satisfy the 2005-2026 full-history VN30 requirement.",
        "- Any selected subset must be labeled as an available-window VN30 subset unless all 30 tickers are selected.",
        "- The selected split is data-driven because the suggested 2025 evaluation start is not feasible for the local hourly universe.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def write_decision_report(path: Path, design: dict[str, Any]) -> None:
    selected = design.get("selected_tickers", [])
    excluded = design.get("excluded_tickers", [])
    exclusion_rows = [
        {"ticker": ticker, "reason": design.get("exclusion_reasons", {}).get(ticker, "")}
        for ticker in excluded
    ]
    content = [
        "# VN30 Hourly Available-Window Design Decision",
        "",
        "## Selected Design",
        "",
        f"- Selected tickers: {', '.join(selected) if selected else 'None'}.",
        f"- Excluded tickers: {', '.join(excluded) if excluded else 'None'}.",
        f"- Training start: {design.get('training_start', '')}.",
        f"- Training cutoff: {design.get('training_cutoff', '')}.",
        f"- Evaluation start: {design.get('evaluation_start', '')}.",
        f"- Evaluation end: {design.get('evaluation_end', '')}.",
        "- Frequency: hourly only.",
        f"- Minimum row count: {design.get('minimum_row_count', 0)} common hourly timestamps per selected ticker window.",
        f"- Claim boundary, full VN30 representativeness: {str(bool(design.get('full_vn30_representativeness'))).lower()}.",
        f"- Final paper can proceed: {str(bool(design.get('final_paper_can_proceed'))).lower()}.",
        "",
        "## Selection Rationale",
        "",
        f"- Selection rule: {design.get('selection_rule', '')}.",
        f"- Minimum split rule: {design.get('minimum_split_rule', '')}.",
        f"- Claim boundary: {design.get('claim_boundary', '')}.",
        "- The suggested 2025-01-01 evaluation start is not feasible for the local hourly universe because most tickers start in 2025 or 2026.",
        "",
        "## Exclusions",
        "",
        markdown_table(["ticker", "reason"], exclusion_rows) if exclusion_rows else "None.",
        "",
        "## Evidence Boundary",
        "",
        "- This is not a 2005-2026 full-history VN30 benchmark.",
        "- Daily data is not used.",
        "- VN100 evidence is not reused.",
        "- No data is fabricated.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def index_audit_rows(design: dict[str, Any]) -> list[dict[str, Any]]:
    window_start_text = str(design.get("training_start", ""))
    train_cutoff_text = str(design.get("training_cutoff", ""))
    eval_start_text = str(design.get("evaluation_start", ""))
    window_end_text = str(design.get("evaluation_end", ""))
    has_design_window = bool(window_start_text and train_cutoff_text and eval_start_text and window_end_text)
    window_start = pd.Timestamp(window_start_text) if has_design_window else None
    train_cutoff = pd.Timestamp(train_cutoff_text) if has_design_window else None
    eval_start = pd.Timestamp(eval_start_text) if has_design_window else None
    window_end = pd.Timestamp(window_end_text) if has_design_window else None
    rows: list[dict[str, Any]] = []
    for index_code in INDEX_CODES:
        frame, sources, files = load_hourly_sources_for_ticker(index_code)
        timestamps = pd.to_datetime(frame["datetime"], errors="coerce").dropna() if not frame.empty else pd.Series(dtype="datetime64[ns]")
        first_ts = pd.Timestamp(timestamps.min()) if not timestamps.empty else None
        last_ts = pd.Timestamp(timestamps.max()) if not timestamps.empty else None
        if has_design_window and not frame.empty:
            selected_rows = int(((frame["datetime"] >= window_start) & (frame["datetime"] <= window_end)).sum())
            selected_train_rows = int(((frame["datetime"] >= window_start) & (frame["datetime"] <= train_cutoff)).sum())
            selected_eval_rows = int(((frame["datetime"] >= eval_start) & (frame["datetime"] <= window_end)).sum())
            overlaps = bool(selected_rows > 0)
            covers = bool(first_ts is not None and last_ts is not None and first_ts <= window_start and last_ts >= window_end)
        else:
            selected_rows = 0
            selected_train_rows = 0
            selected_eval_rows = 0
            overlaps = False
            covers = False
        exact_found = bool(not frame.empty)
        ready = bool(exact_found and overlaps and covers and selected_train_rows > 0 and selected_eval_rows > 0)
        if not has_design_window:
            limitation = "selected_available_window_design_missing"
        elif not exact_found:
            limitation = "exact_code_local_hourly_index_data_missing; context_not_fabricated"
        elif not overlaps:
            limitation = "exact_code_local_hourly_index_data_does_not_overlap_selected_window"
        elif not covers:
            limitation = "exact_code_local_hourly_index_data_does_not_cover_selected_window"
        elif selected_train_rows == 0 or selected_eval_rows == 0:
            limitation = "selected_train_or_eval_index_rows_missing"
        else:
            limitation = "usable_for_available_window_context"
        rows.append(
            {
                "index_code": index_code,
                "exact_code_local_file_found": exact_found,
                "first_available_hourly_timestamp": timestamp_text(first_ts),
                "last_available_hourly_timestamp": timestamp_text(last_ts),
                "hourly_rows": int(len(frame)),
                "selected_window_start": window_start_text,
                "selected_window_end": window_end_text,
                "selected_window_rows": selected_rows,
                "selected_train_rows": selected_train_rows,
                "selected_eval_rows": selected_eval_rows,
                "overlaps_selected_window": overlaps,
                "covers_selected_window": covers,
                "available_window_context_ready": ready,
                "limitation": limitation,
                "raw_hourly_sources": ";".join(sources),
                "raw_hourly_files": ";".join(files),
            }
        )
    return rows


def write_index_audit_report(path: Path, rows: list[dict[str, Any]], design: dict[str, Any]) -> None:
    ready_count = sum(1 for row in rows if bool(row.get("available_window_context_ready")))
    selected_window = f"{design.get('training_start', '')} to {design.get('evaluation_end', '')}"
    content = [
        "# VN30 Hourly Available-Window Market Index Audit",
        "",
        "## Scope",
        "",
        "- Exact local hourly index codes audited: `VNINDEX`, `VN30INDEX`, `VNXALL`.",
        "- No aliases are used. Local stock symbols such as `VNI` or `VNX` are not treated as market indices.",
        "- Daily data, daily-to-hourly resampling, and fabricated index context are not used.",
        f"- Selected stock available-window period: {selected_window}.",
        "",
        "## Result",
        "",
        f"- Available-window index context ready: {ready_count} of 3 exact codes.",
        "- If exact local index data is missing, this is a limitation rather than a reason to fabricate context.",
        "",
        "## Per-Index Audit",
        "",
        markdown_table(
            [
                "index_code",
                "exact_code_local_file_found",
                "first_available_hourly_timestamp",
                "last_available_hourly_timestamp",
                "selected_window_rows",
                "selected_train_rows",
                "selected_eval_rows",
                "overlaps_selected_window",
                "covers_selected_window",
                "available_window_context_ready",
                "limitation",
            ],
            rows,
        ),
        "",
        "## Boundary",
        "",
        "- The available-window stock benchmark remains real local hourly evidence.",
        "- Missing exact-code local index data must be disclosed as a market-context limitation.",
        "- Do not reconstruct pre-start or missing index history unless the vendor supplies it and clearly labels it.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> int:
    args = parse_args()
    tickers = read_universe()
    if tickers != VN30_TICKERS:
        raise ValueError("Frozen VN30 universe file does not match the mandatory 30-ticker list.")
    rows, frames = ticker_audit_rows()
    threshold_rows, feasible_candidates = coverage_windows(rows, frames)
    design = select_design(rows, threshold_rows, feasible_candidates)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = args.output_dir / "vn30_hourly_available_window_audit.csv"
    audit_md = args.output_dir / "vn30_hourly_available_window_audit.md"
    windows_csv = args.output_dir / "vn30_hourly_available_window_coverage_windows.csv"
    write_csv(audit_csv, rows, fieldnames=AUDIT_COLUMNS)
    write_csv(windows_csv, threshold_rows, fieldnames=WINDOW_COLUMNS)
    write_audit_report(audit_md, rows, threshold_rows, design)
    index_rows = index_audit_rows(design)
    index_csv = args.output_dir / "vn30_hourly_available_window_index_audit.csv"
    index_md = args.output_dir / "vn30_hourly_available_window_index_audit.md"
    write_csv(index_csv, index_rows, fieldnames=INDEX_AUDIT_COLUMNS)
    write_index_audit_report(index_md, index_rows, design)
    write_json(args.decision_json, design)
    write_decision_report(args.decision_md, design)
    write_docx_build_notes(DOCX_NOTES_PATH, design)
    print(
        "VN30 hourly available-window audit complete: "
        f"selected={len(design.get('selected_tickers', []))}/30 "
        f"paper={str(bool(design.get('final_paper_can_proceed'))).lower()} "
        f"audit={rel(audit_md)} decision={rel(args.decision_md)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
