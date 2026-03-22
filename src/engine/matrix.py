"""Module 1: The Decision Matrix

Consolidates Quantitative signals (ML) and Qualitative analysis (LLM)
to generate the final execution matrix consensus based on hard rules.
"""

from __future__ import annotations

from src.api.schemas import MatrixConsensus, QualitativeAnalysis, QuantitativeSignals


def evaluate_decision_matrix(
    quant: QuantitativeSignals,
    qual: QualitativeAnalysis | None = None,
) -> tuple[str, MatrixConsensus]:
    """Evaluate signals through the Rule Engine.

    Rules:
        Rule 1 (Perfect Match): BUY + POSITIVE -> EXECUTE_BUY
                                SELL + NEGATIVE -> EXECUTE_SELL
        Rule 2 (Veto Rule): BUY + NEGATIVE -> CANCEL_ORDER
                            SELL + POSITIVE -> CANCEL_ORDER
        Rule 3 (Null Rule): Missing LLM or insufficient data -> STANDBY

    Args:
        quant: Phase 2 ML quantitative output.
        qual: Phase 3 LLM qualitative output (if any).

    Returns:
        tuple containing:
            - string: execution decision (e.g., EXECUTE_BUY, STANDBY, CANCEL_ORDER)
            - MatrixConsensus model
    """
    ml_rec = quant.action_plan.recommendation  # BUY, SELL, RANGE_TRADE, STAND_ASIDE

    # Rule 3: Null Rule - missing data or insufficient data
    if not qual or qual.analysis_status != "success":
        return "STANDBY", MatrixConsensus(
            ml_signal=ml_rec,
            llm_sentiment="N/A" if not qual else qual.sentiment,
            veto_triggered=False,
        )

    llm_sentiment = qual.sentiment.upper()  # POSITIVE, NEGATIVE, NEUTRAL

    # Handle Neutral LLM sentiment with Neutral ML
    if ml_rec in ("RANGE_TRADE", "STAND_ASIDE"):
        return "STANDBY", MatrixConsensus(
            ml_signal=ml_rec,
            llm_sentiment=llm_sentiment,
            veto_triggered=False,
        )

    # Rule 1 & 2 logic execution
    decision = "STANDBY"
    veto = False

    if ml_rec == "BUY":
        if llm_sentiment == "POSITIVE":
            decision = "EXECUTE_BUY"
        elif llm_sentiment == "NEGATIVE":
            decision = "CANCEL_ORDER"
            veto = True
        else:
            decision = "STANDBY"

    elif ml_rec == "SELL":
        if llm_sentiment == "NEGATIVE":
            decision = "EXECUTE_SELL"
        elif llm_sentiment == "POSITIVE":
            decision = "CANCEL_ORDER"
            veto = True
        else:
            decision = "STANDBY"

    consensus = MatrixConsensus(
        ml_signal=ml_rec,
        llm_sentiment=llm_sentiment,
        veto_triggered=veto,
    )

    return decision, consensus
