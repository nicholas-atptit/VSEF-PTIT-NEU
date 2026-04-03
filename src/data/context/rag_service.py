"""Zone-aware RAG querying backed by ChromaDB metadata filters."""

from __future__ import annotations

import datetime as dt
from typing import Any

from config.settings import get_settings
from src.data.context.embedder import DocumentEmbedder
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ZonedRAGService:
    """Query ChromaDB with strict zone, ticker, and time filters."""

    ZONE_LABELS = {
        "zone_1": "BCTC kiem toan",
        "zone_2": "Bao cao phan tich CTCK",
        "zone_3": "Tin tuc chinh thong",
        "zone_4": "Tin don va sentiment",
    }
    ZONE_DOC_TYPES = {
        "zone_1": "report",
        "zone_2": "analysis",
        "zone_3": "news",
        "zone_4": "news",
    }
    ZONE_QUERY_HINTS = {
        "zone_1": "Bao cao tai chinh kiem toan, doanh thu, loi nhuan, dong tien cua {ticker}",
        "zone_2": "Bao cao phan tich chung khoan va dinh gia doanh nghiep {ticker}",
        "zone_3": "Tin tuc doanh nghiep, vi mo va su kien anh huong den co phieu {ticker}",
        "zone_4": "Tam ly thi truong, dong tien dau co va tin don lien quan den {ticker}",
    }

    def __init__(self) -> None:
        self._settings = get_settings()
        self._embedder = DocumentEmbedder()

    def query(
        self,
        ticker: str,
        allowed_zones: list[str],
        query_text: str | None = None,
        n_results: int | None = None,
        as_of: dt.datetime | None = None,
        horizon: str = "short", # short, mid, long
    ) -> str:
        """Return formatted context text ready for LLM injection."""
        normalized_ticker = ticker.upper().strip()
        cutoff_dt = self._normalize_datetime(as_of)
        results_by_zone = self.query_documents(
            ticker=normalized_ticker,
            allowed_zones=allowed_zones,
            query_text=query_text,
            n_results=n_results,
            as_of=cutoff_dt,
            horizon=horizon,
        )

        if not results_by_zone:
            return f"[rag] Khong co zone hop le cho {normalized_ticker}."

        context_parts = [
            self._format_zone_context(zone=zone, matches=results_by_zone.get(zone, []), cutoff_dt=cutoff_dt)
            for zone in results_by_zone
        ]
        rag_text = "\n\n".join(context_parts)
        logger.info(
            "rag_query_done",
            ticker=normalized_ticker,
            zones=list(results_by_zone.keys()),
            total_matches=sum(len(matches) for matches in results_by_zone.values()),
        )
        return rag_text

    def query_documents(
        self,
        ticker: str,
        allowed_zones: list[str],
        query_text: str | None = None,
        n_results: int | None = None,
        as_of: dt.datetime | None = None,
        horizon: str = "short",
    ) -> dict[str, list[dict[str, Any]]]:
        """Return raw ChromaDB matches grouped by zone."""
        normalized_ticker = ticker.upper().strip()
        normalized_zones = [zone for zone in allowed_zones if zone in self.ZONE_LABELS]
        if not normalized_zones:
            return {}

        cutoff_dt = self._normalize_datetime(as_of)
        effective_top_k = n_results or self._settings.rag_top_k_per_zone
        
        # Horizon-aware lookback
        lookback_map = {
            "short": self._settings.short_horizon_days or 7,
            "mid": self._settings.mid_horizon_days or 30,
            "long": self._settings.long_horizon_days or 180
        }
        lookback_days = lookback_map.get(horizon.lower(), self._settings.rag_news_lookback_days)
        
        lookback_dt = cutoff_dt - dt.timedelta(days=lookback_days)
        results_by_zone: dict[str, list[dict[str, Any]]] = {}

        for zone in normalized_zones:
            doc_type = self.ZONE_DOC_TYPES[zone]
            zone_query = query_text or self.ZONE_QUERY_HINTS[zone].format(ticker=normalized_ticker)
            published_after = lookback_dt if doc_type == "news" else None

            try:
                results_by_zone[zone] = self._embedder.search(
                    query=zone_query,
                    doc_type=doc_type,
                    n_results=effective_top_k,
                    ticker=normalized_ticker,
                    zone=zone,
                    published_before=cutoff_dt,
                    published_after=published_after,
                )
            except Exception as exc:
                logger.warning(
                    "rag_zone_query_error",
                    ticker=normalized_ticker,
                    zone=zone,
                    error=str(exc),
                )
                results_by_zone[zone] = []

        return results_by_zone

    def _format_zone_context(
        self,
        zone: str,
        matches: list[dict[str, Any]],
        cutoff_dt: dt.datetime,
    ) -> str:
        """Format a zone block for prompt injection."""
        zone_label = self.ZONE_LABELS.get(zone, zone)
        header = f"[{zone}] {zone_label} (cutoff={cutoff_dt.date().isoformat()}):"
        if not matches:
            return f"{header}\n- Khong tim thay tai lieu phu hop trong ChromaDB."

        lines = [header]
        for match in matches:
            metadata = match.get("metadata", {})
            title = metadata.get("title") or metadata.get("source") or "Tai lieu khong ten"
            source = metadata.get("source", "")
            published_date = str(metadata.get("published_date", ""))[:10]
            text = " ".join(str(match.get("text", "")).split())
            if len(text) > 320:
                text = f"{text[:317].rstrip()}..."

            descriptor_parts = [part for part in [source, published_date] if part]
            descriptor = f" ({', '.join(descriptor_parts)})" if descriptor_parts else ""
            lines.append(f"- {title}{descriptor}: {text}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_datetime(value: dt.datetime | None) -> dt.datetime:
        """Normalize datetimes to aware UTC values."""
        if value is None:
            return dt.datetime.now(dt.UTC)
        if value.tzinfo is None:
            return value.replace(tzinfo=dt.UTC)
        return value.astimezone(dt.UTC)
