"""Normalization and generation layer for the Analysis Feed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.core.analysis_schema import (
    ForecastResearchPacket,
    HistoricalCaseRecord,
    AnalystMemoDraft,
    RetrievalMetadata,
    AnalysisFeedManifest
)
from src.reporting.case_records import (
    build_deterministic_summary,
    attach_realized_outcomes,
    build_case_tags
)


def load_quant_core_manifest(quant_core_dir: str | Path) -> Dict[str, Any]:
    """Load the source quant-core manifest."""
    manifest_path = Path(quant_core_dir) / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Quant-core manifest not found at {manifest_path}")
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _generate_packet_id(run_id: str, ticker: str, date: str, horizon: int, target: str, mode: str) -> str:
    return f"pkt_{run_id}_{ticker}_{date}_h{horizon:02d}_{target}_{mode}"


def normalize_to_research_packets(
    quant_core_dir: str | Path,
    manifest: Dict[str, Any]
) -> List[ForecastResearchPacket]:
    """Transform analysis_packets.jsonl into ForecastResearchPacket list."""
    
    packet_file = Path(manifest["artifact_paths"]["analysis_packets"])
    if not packet_file.is_absolute():
        packet_file = Path(quant_core_dir) / packet_file.name

    run_id = manifest["git"]["commit_hash"][:8] # Using prefix as run_id if not explicit
    # Check if core_run_id is in manifest meta, else fallback to commit
    core_run_id = manifest.get("core_run_id") or run_id
    
    source_manifest_path = str(Path(quant_core_dir) / "run_manifest.json")
    source_artifact_dir = str(quant_core_dir)

    packets: List[ForecastResearchPacket] = []
    
    with open(packet_file, "r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            
            # Stable ID generation
            # Usually packet has 'timestamp' and 'ticker'
            date_str = str(pd.Timestamp(raw["timestamp"]).date())
            packet_id = _generate_packet_id(
                core_run_id,
                raw["ticker"],
                date_str,
                raw["horizon"],
                raw["target_type"],
                raw["run_mode"]
            )
            
            # Mapping fields
            packet = ForecastResearchPacket(
                source_run_id=core_run_id,
                source_manifest_path=source_manifest_path,
                source_artifact_dir=source_artifact_dir,
                packet_id=packet_id,
                timestamp=raw["timestamp"],
                ticker=raw["ticker"],
                ticker_group=raw.get("group_name"),
                horizon=raw["horizon"],
                target_type=raw["target_type"],
                run_mode=raw["run_mode"],
                cost_mode=raw.get("cost_mode"),
                primary_model_name=raw["primary_model_name"],
                primary_model_role=raw["primary_model_role"],
                primary_prediction=raw.get("primary_prediction"),
                model_by_model_predictions=json.loads(raw["model_by_model_predictions"]) if isinstance(raw["model_by_model_predictions"], str) else raw["model_by_model_predictions"],
                model_ranks=json.loads(raw["model_ranks"]) if isinstance(raw["model_ranks"], str) else raw["model_ranks"],
                ensemble_summary=json.loads(raw["ensemble_summary"]) if isinstance(raw["ensemble_summary"], str) and raw["ensemble_summary"] else (raw["ensemble_summary"] or None),
                model_agreement_score=raw.get("model_agreement_score"),
                model_disagreement_score=raw.get("model_disagreement_score"),
                dispersion_score=raw.get("dispersion_score"),
                sign_conflict=raw.get("sign_conflict", False),
                rank_spread=raw.get("rank_spread"),
                agreement_bucket=raw.get("agreement_bucket", "unknown"),
                risk_summary=json.loads(raw["risk_summary"]) if isinstance(raw["risk_summary"], str) and raw["risk_summary"] else (raw["risk_summary"] or None),
                regime_summary=json.loads(raw["regime_summary"]) if isinstance(raw["regime_summary"], str) and raw["regime_summary"] else (raw["regime_summary"] or None),
                policy_summary=json.loads(raw["policy_summary"]) if isinstance(raw["policy_summary"], str) and raw["policy_summary"] else (raw["policy_summary"] or None),
                policy_gate_disagreement_share=raw.get("policy_gate_disagreement_share"),
                volatility_bucket=raw.get("volatility_bucket", "unknown"),
                signal_strength_bucket=raw.get("signal_strength_bucket", "unknown"),
                # realized_y_true from packet maps to outcome
                **attach_realized_outcomes(raw)
            )
            packets.append(packet)
            
    return packets


def generate_case_records(packets: List[ForecastResearchPacket]) -> List[HistoricalCaseRecord]:
    """Convert packets into retrieval-ready Case Records."""
    cases: List[HistoricalCaseRecord] = []
    for pkt in packets:
        case = HistoricalCaseRecord(
            source_run_id=pkt.source_run_id,
            source_manifest_path=pkt.source_manifest_path,
            source_artifact_dir=pkt.source_artifact_dir,
            case_id=f"case_{pkt.packet_id}",
            source_packet_id=pkt.packet_id,
            ticker=pkt.ticker,
            ticker_group=pkt.ticker_group,
            horizon=pkt.horizon,
            target_type=pkt.target_type,
            regime_label=pkt.regime_summary.get("regime_label") if pkt.regime_summary else None,
            volatility_bucket=pkt.volatility_bucket,
            signal_strength_bucket=pkt.signal_strength_bucket,
            agreement_bucket=pkt.agreement_bucket,
            model_role_context=pkt.primary_model_role, # Simplified
            summary_text=build_deterministic_summary(pkt.model_dump()),
            run_mode=pkt.run_mode,
            timestamp=pkt.timestamp,
            realized_outcome_label=pkt.realized_outcome_label,
            tags=build_case_tags(pkt.model_dump())
        )
        cases.append(case)
    return cases


def generate_memo_drafts(packets: List[ForecastResearchPacket]) -> List[AnalystMemoDraft]:
    """Generate structured memo drafts with deterministic placeholders."""
    memos: List[AnalystMemoDraft] = []
    for pkt in packets:
        # We only really care about memos for 'primary_research' or candidates, 
        # but the plan says generate from packets.
        memo = AnalystMemoDraft(
            source_run_id=pkt.source_run_id,
            source_manifest_path=pkt.source_manifest_path,
            source_artifact_dir=pkt.source_artifact_dir,
            memo_id=f"memo_{pkt.packet_id}",
            related_packet_id=pkt.packet_id,
            ticker=pkt.ticker,
            horizon=pkt.horizon,
            ticker_snapshot={
                "ticker": pkt.ticker,
                "group": pkt.ticker_group,
                "regime": pkt.regime_summary.get("regime_label") if pkt.regime_summary else "unknown",
                "volatility": pkt.volatility_bucket
            },
            primary_signal_summary=f"Model {pkt.primary_model_name} predicts {pkt.primary_prediction:.4f} ({pkt.signal_strength_bucket} strength)",
            model_consensus_summary=f"Agreement: {pkt.agreement_bucket} ({pkt.model_agreement_score:.2f})",
            model_conflicts=["Sign conflict detected"] if pkt.sign_conflict else [],
            regime_summary=f"Current regime: {pkt.regime_summary.get('regime_label')}" if pkt.regime_summary else "Regime unavailable",
            risk_summary=f"Vol: {pkt.risk_summary.get('vol_forecast'):.4f}" if pkt.risk_summary and pkt.risk_summary.get('vol_forecast') else "Risk unavailable",
            suggested_action_candidate="Review for entry" if pkt.agreement_bucket in ["medium", "high"] and (pkt.primary_prediction or 0) > 0.01 else "Monitor"
        )
        memos.append(memo)
    return memos


def generate_retrieval_metadata(
    packets: List[ForecastResearchPacket],
    cases: List[HistoricalCaseRecord],
    memos: List[AnalystMemoDraft]
) -> List[RetrievalMetadata]:
    """Generate flat metadata rows for CSV/Indexing."""
    metadata_rows: List[RetrievalMetadata] = []
    # Alignment check
    for pkt, case, memo in zip(packets, cases, memos):
        meta = RetrievalMetadata(
            related_packet_id=pkt.packet_id,
            related_case_id=case.case_id,
            related_memo_id=memo.memo_id,
            ticker=pkt.ticker,
            ticker_group=pkt.ticker_group,
            horizon=pkt.horizon,
            target_type=pkt.target_type,
            regime_label=case.regime_label,
            volatility_bucket=pkt.volatility_bucket,
            signal_strength_bucket=pkt.signal_strength_bucket,
            agreement_bucket=pkt.agreement_bucket,
            model_role_context=pkt.primary_model_role,
            run_mode=pkt.run_mode,
            cost_mode=pkt.cost_mode,
            date_range=pkt.timestamp # Snapshot
        )
        metadata_rows.append(meta)
    return metadata_rows


def write_analysis_feed(
    output_dir: str | Path,
    manifest: Dict[str, Any],
    packets: List[ForecastResearchPacket],
    cases: List[HistoricalCaseRecord],
    memos: List[AnalystMemoDraft],
    metadata_rows: List[RetrievalMetadata]
) -> Dict[str, str]:
    """Write all feed artifacts and return their relative paths."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Packets JSONL
    pkt_file = out_path / "forecast_research_packets.jsonl"
    with open(pkt_file, "w", encoding="utf-8") as f:
        for pkt in packets:
            f.write(pkt.model_dump_json() + "\n")
    paths["forecast_research_packets"] = str(pkt_file.relative_to(out_path))
    
    # Cases JSONL
    case_file = out_path / "historical_case_records.jsonl"
    with open(case_file, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(case.model_dump_json() + "\n")
    paths["historical_case_records"] = str(case_file.relative_to(out_path))
    
    # Memos JSONL
    memo_file = out_path / "analyst_memo_drafts.jsonl"
    with open(memo_file, "w", encoding="utf-8") as f:
        for memo in memos:
            f.write(memo.model_dump_json() + "\n")
    paths["analyst_memo_drafts"] = str(memo_file.relative_to(out_path))
    
    # Metadata CSV
    meta_df = pd.DataFrame([m.model_dump() for m in metadata_rows])
    meta_file = out_path / "retrieval_metadata.csv"
    meta_df.to_csv(meta_file, index=False)
    paths["retrieval_metadata"] = str(meta_file.relative_to(out_path))
    
    # Queue CSV (Deterministic Filter)
    queue_df = meta_df[
        (meta_df["agreement_bucket"].isin(["medium", "high"])) &
        (meta_df["signal_strength_bucket"].isin(["medium", "high"]))
    ].copy()
    queue_file = out_path / "candidate_review_queue.csv"
    queue_df.to_csv(queue_file, index=False)
    paths["candidate_review_queue"] = str(queue_file.relative_to(out_path))
    
    return paths
