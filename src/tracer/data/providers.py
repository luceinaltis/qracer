"""Data layer protocols.

Each provider protocol defines a capability that data adapters can implement.
The DataRegistry routes requests by capability with fallback support.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class OHLCV:
    """A single OHLCV bar."""

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class FundamentalData:
    """Fundamental data snapshot for a security."""

    ticker: str
    pe_ratio: float | None = None
    market_cap: float | None = None
    revenue: float | None = None
    earnings: float | None = None
    dividend_yield: float | None = None
    fetched_at: datetime | None = None


@dataclass(frozen=True)
class MacroIndicator:
    """A macroeconomic indicator data point."""

    name: str
    value: float
    date: date
    source: str
    unit: str = ""


@dataclass(frozen=True)
class NewsArticle:
    """A news article with optional sentiment score."""

    title: str
    source: str
    published_at: datetime
    url: str
    summary: str = ""
    sentiment: float | None = None  # -1.0 to 1.0


@dataclass(frozen=True)
class AlternativeRecord:
    """Alternative data record (insider trades, congressional trades, etc.)."""

    record_type: str
    ticker: str
    data: dict
    source: str
    date: date


@runtime_checkable
class PriceProvider(Protocol):
    """Capability: Price/OHLCV data retrieval."""

    async def get_price(self, ticker: str) -> float: ...

    async def get_ohlcv(self, ticker: str, start: date, end: date) -> list[OHLCV]: ...


@runtime_checkable
class FundamentalProvider(Protocol):
    """Capability: Fundamental financial data retrieval."""

    async def get_fundamentals(self, ticker: str) -> FundamentalData: ...


@runtime_checkable
class MacroProvider(Protocol):
    """Capability: Macroeconomic indicator retrieval."""

    async def get_indicator(self, name: str) -> MacroIndicator: ...


@runtime_checkable
class NewsProvider(Protocol):
    """Capability: News and sentiment data retrieval."""

    async def get_news(self, ticker: str, limit: int = 10) -> list[NewsArticle]: ...


@runtime_checkable
class AlternativeProvider(Protocol):
    """Capability: Alternative data retrieval (insider trades, etc.)."""

    async def get_alternative(self, ticker: str, record_type: str) -> list[AlternativeRecord]: ...
