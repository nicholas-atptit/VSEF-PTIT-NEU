"""Loader for audited financial statements (Zone 1) from PDF and Excel files."""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from pathlib import Path
from typing import Any

from config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BCTCLoader:
    """Load and normalize BCTC files into embedding-ready documents."""

    SUPPORTED_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".csv"}

    def __init__(self, data_dir: str | None = None) -> None:
        settings = get_settings()
        self._data_dir = Path(data_dir or settings.bctc_data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def load_directory(
        self,
        ticker: str | None = None,
        max_files: int = 50,
    ) -> list[dict[str, Any]]:
        """Load a directory of BCTC files filtered by ticker when provided."""
        ticker_filter = ticker.upper().strip() if ticker else None
        candidates: list[Path] = []

        for file_path in sorted(self._data_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            if ticker_filter and ticker_filter not in file_path.stem.upper():
                continue
            candidates.append(file_path)

        documents: list[dict[str, Any]] = []
        for file_path in candidates[:max_files]:
            try:
                doc = self._load_file(file_path, ticker=ticker_filter)
                if doc and len(doc.get("content", "")) > 100:
                    documents.append(doc)
                    logger.info(
                        "bctc_loaded",
                        file=file_path.name,
                        ticker=doc.get("primary_ticker"),
                        chars=len(doc["content"]),
                    )
            except Exception as exc:
                logger.error("bctc_load_error", file=str(file_path), error=str(exc))

        logger.info(
            "bctc_directory_scanned",
            dir=str(self._data_dir),
            total=len(documents),
            candidates=len(candidates),
        )
        return documents

    def load_file(self, filepath: str, ticker: str = "") -> dict[str, Any] | None:
        """Load a single BCTC file."""
        return self._load_file(Path(filepath), ticker=ticker.upper().strip() or None)

    def _load_file(self, file_path: Path, ticker: str | None = None) -> dict[str, Any] | None:
        """Parse a file based on extension and attach Zone 1 metadata."""
        extension = file_path.suffix.lower()
        if extension == ".pdf":
            content = self._parse_pdf(file_path)
        elif extension in (".xlsx", ".xls"):
            content = self._parse_excel(file_path)
        elif extension == ".csv":
            content = self._parse_csv(file_path)
        else:
            return None

        content = self._clean_text(content)
        if not content:
            return None

        detected_ticker = ticker or self._detect_ticker_from_filename(file_path.stem)
        report_year, report_quarter, report_period = self._detect_period_from_filename(file_path.stem)
        published_date = self._detect_date_from_filename(file_path.stem)
        doc_id = hashlib.sha256(str(file_path.resolve()).encode()).hexdigest()[:16]

        return {
            "doc_id": doc_id,
            "title": f"BCTC - {file_path.stem}",
            "content": content,
            "published_date": published_date,
            "source": "bctc_audit",
            "source_type": "local_file",
            "tickers": detected_ticker,
            "primary_ticker": detected_ticker,
            "url": str(file_path.resolve()),
            "doc_type": "report",
            "zone": "zone_1",
            "report_period": report_period,
            "report_year": report_year or "",
            "report_quarter": report_quarter,
            "file_extension": extension.lstrip("."),
        }

    def _parse_pdf(self, file_path: Path) -> str:
        """Extract text and basic table rows from a PDF."""
        try:
            import pdfplumber

            text_parts: list[str] = []
            with pdfplumber.open(str(file_path)) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_parts.append(f"[Page {page_number}]\n{page_text}")
                    for table in page.extract_tables() or []:
                        for row in table:
                            cleaned_row = [str(cell).strip() for cell in row or [] if cell not in (None, "")]
                            if cleaned_row:
                                text_parts.append(" | ".join(cleaned_row))
            return "\n\n".join(text_parts)
        except ImportError:
            logger.warning("pdfplumber_unavailable", file=str(file_path))
        except Exception as exc:
            logger.warning("pdfplumber_parse_error", file=str(file_path), error=str(exc))

        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(file_path))
            text_parts = []
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(f"[Page {page_number}]\n{page_text}")
            return "\n\n".join(text_parts)
        except ImportError:
            logger.warning("pypdf2_unavailable", file=str(file_path))
        except Exception as exc:
            logger.error("pdf_parse_error", file=str(file_path), error=str(exc))

        return ""

    def _parse_excel(self, file_path: Path) -> str:
        """Extract text from all non-empty sheets of an Excel workbook."""
        import pandas as pd

        text_parts: list[str] = []
        try:
            workbook = pd.ExcelFile(str(file_path))
            for sheet_name in workbook.sheet_names:
                dataframe = pd.read_excel(workbook, sheet_name=sheet_name, dtype=str)
                sheet_text = self._dataframe_to_text(dataframe)
                if sheet_text:
                    text_parts.append(f"=== Sheet: {sheet_name} ===")
                    text_parts.append(sheet_text)
        except Exception as exc:
            logger.error("excel_parse_error", file=str(file_path), error=str(exc))

        return "\n\n".join(text_parts)

    def _parse_csv(self, file_path: Path) -> str:
        """Extract text from CSV files."""
        import pandas as pd

        try:
            dataframe = pd.read_csv(str(file_path), dtype=str)
            csv_text = self._dataframe_to_text(dataframe)
            if csv_text:
                return f"=== {file_path.stem} ===\n{csv_text}"
            return ""
        except Exception as exc:
            logger.error("csv_parse_error", file=str(file_path), error=str(exc))
            return ""

    @staticmethod
    def _dataframe_to_text(dataframe: Any) -> str:
        """Convert a dataframe to dense plain text suitable for embeddings."""
        if dataframe is None:
            return ""

        dataframe = dataframe.dropna(how="all").dropna(axis=1, how="all")
        if dataframe.empty:
            return ""

        dataframe = dataframe.fillna("")
        header = " | ".join(str(column).strip() for column in dataframe.columns)
        rows = []
        for _, row in dataframe.iterrows():
            values = [str(value).strip() for value in row.tolist()]
            if any(values):
                rows.append(" | ".join(values))

        return "\n".join([header, *rows]).strip()

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace before embedding."""
        if not text:
            return ""

        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _detect_ticker_from_filename(stem: str) -> str:
        """Extract a 3-letter ticker from the filename when available."""
        match = re.search(r"(?:^|[^A-Z])([A-Z]{3})(?:[^A-Z]|$)", stem.upper())
        return match.group(1) if match else ""

    @staticmethod
    def _detect_period_from_filename(stem: str) -> tuple[int | None, str, str]:
        """Extract year and quarter markers from filenames like SSI_Q3_2024_BCTC."""
        normalized = stem.upper()
        year_match = re.search(r"(20\d{2})", normalized)
        quarter_match = re.search(
            r"(?:^|[_\-\s])Q([1-4])(?:$|[_\-\s])|(?:^|[_\-\s])QUY[_\-\s]?([1-4])(?:$|[_\-\s])",
            normalized,
        )

        year = int(year_match.group(1)) if year_match else None
        quarter_number = ""
        if quarter_match:
            quarter_number = quarter_match.group(1) or quarter_match.group(2) or ""

        report_quarter = f"Q{quarter_number}" if quarter_number else ""
        if report_quarter and year:
            report_period = f"{report_quarter}-{year}"
        elif year:
            report_period = f"FY{year}"
        else:
            report_period = ""

        return year, report_quarter, report_period

    def _detect_date_from_filename(self, stem: str) -> str:
        """Infer a canonical date from the report period in the filename."""
        year, report_quarter, _ = self._detect_period_from_filename(stem)
        if year is None:
            return ""

        if report_quarter == "Q1":
            return f"{year}-03-31"
        if report_quarter == "Q2":
            return f"{year}-06-30"
        if report_quarter == "Q3":
            return f"{year}-09-30"
        if report_quarter == "Q4":
            return f"{year}-12-31"
        return f"{year}-12-31"
