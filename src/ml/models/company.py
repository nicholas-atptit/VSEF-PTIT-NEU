"""Company Profile models for fundamental data."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import BigInteger, Boolean, Date, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.ml.models.base import Base, TimestampMixin


class CompanyProfile(Base, TimestampMixin):
    """General company information and fundamentals."""
    __tablename__ = "company_profiles"

    ticker: Mapped[str] = mapped_column(String(10), primary_key=True)
    exchange: Mapped[str] = mapped_column(String(10), index=True)
    industry: Mapped[str | None] = mapped_column(String(100), index=True)
    company_name: Mapped[str | None] = mapped_column(String(255))
    short_name: Mapped[str | None] = mapped_column(String(100))
    
    # Fundamentals
    issue_share: Mapped[float | None] = mapped_column(Float, comment="Outstanding shares")
    charter_capital: Mapped[float | None] = mapped_column(Float, comment="Charter capital")
    market_cap: Mapped[float | None] = mapped_column(Float, comment="Market capitalization")
    
    # Additional Context
    established_year: Mapped[str | None] = mapped_column(String(10))
    no_employees: Mapped[int | None] = mapped_column(BigInteger)
    no_shareholders: Mapped[int | None] = mapped_column(BigInteger)
    foreign_percent: Mapped[float | None] = mapped_column(Float, comment="Foreign ownership %")
    
    website: Mapped[str | None] = mapped_column(String(255))
    company_profile: Mapped[str | None] = mapped_column(Text)
    history_dev: Mapped[str | None] = mapped_column(Text)
    company_promise: Mapped[str | None] = mapped_column(Text)
    business_risk: Mapped[str | None] = mapped_column(Text)
    key_developments: Mapped[str | None] = mapped_column(Text)
    business_strategies: Mapped[str | None] = mapped_column(Text)

    def __repr__(self) -> str:
        return f"<CompanyProfile(ticker={self.ticker}, industry={self.industry})>"
