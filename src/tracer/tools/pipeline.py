"""Pipeline tool wrappers — thin async functions that fetch data and return ToolResult.

Each tool wraps a DataRegistry capability lookup, calls the provider, and
normalises the output into the uniform ToolResult contract consumed by the
AnalysisLoop.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from tracer.data.providers import (
    AlternativeProvider,
    FundamentalProvider,
    MacroProvider,
    NewsProvider,
    PriceProvider,
)
from tracer.data.registry import DataRegistry
from tracer.models import ToolResult

logger = logging.getLogger(__name__)

# How many days of price history to fetch by default.
_DEFAULT_LOOKBACK_DAYS = 30
# Default staleness threshold (hours).
_STALENESS_HOURS = 24


def _is_stale(fetched_at: datetime, threshold_hours: int = _STALENESS_HOURS) -> bool:
    return (datetime.now() - fetched_at).total_seconds() > threshold_hours * 3600


async def price_event(ticker: str, registry: DataRegistry) -> ToolResult:
    """Fetch recent price/OHLCV data for a ticker."""
    try:
        provider: PriceProvider = registry.get(PriceProvider)
        end = date.today()
        start = end - timedelta(days=_DEFAULT_LOOKBACK_DAYS)
        bars = await provider.get_ohlcv(ticker, start, end)
        current_price = await provider.get_price(ticker)
        now = datetime.now()
        return ToolResult(
            tool="price_event",
            success=True,
            data={
                "ticker": ticker,
                "current_price": current_price,
                "bars": len(bars),
                "period_start": start.isoformat(),
                "period_end": end.isoformat(),
                "ohlcv": [
                    {
                        "date": b.date.isoformat(),
                        "open": b.open,
                        "high": b.high,
                        "low": b.low,
                        "close": b.close,
                        "volume": b.volume,
                    }
                    for b in bars
                ],
            },
            source="PriceProvider",
            fetched_at=now,
            is_stale=_is_stale(now),
        )
    except Exception as exc:
        logger.warning("price_event failed for %s: %s", ticker, exc)
        return ToolResult(
            tool="price_event",
            success=False,
            data={},
            source="PriceProvider",
            error=str(exc),
        )


async def news(ticker: str, registry: DataRegistry, limit: int = 10) -> ToolResult:
    """Fetch recent news articles for a ticker."""
    try:
        provider: NewsProvider = registry.get(NewsProvider)
        articles = await provider.get_news(ticker, limit=limit)
        now = datetime.now()
        return ToolResult(
            tool="news",
            success=True,
            data={
                "ticker": ticker,
                "count": len(articles),
                "articles": [
                    {
                        "title": a.title,
                        "source": a.source,
                        "published_at": a.published_at.isoformat(),
                        "url": a.url,
                        "summary": a.summary,
                        "sentiment": a.sentiment,
                    }
                    for a in articles
                ],
            },
            source="NewsProvider",
            fetched_at=now,
            is_stale=_is_stale(now),
        )
    except Exception as exc:
        logger.warning("news failed for %s: %s", ticker, exc)
        return ToolResult(
            tool="news", success=False, data={}, source="NewsProvider", error=str(exc)
        )


async def insider(ticker: str, registry: DataRegistry) -> ToolResult:
    """Fetch insider trading data for a ticker."""
    try:
        provider: AlternativeProvider = registry.get(AlternativeProvider)
        records = await provider.get_alternative(ticker, record_type="insider_trades")
        now = datetime.now()
        return ToolResult(
            tool="insider",
            success=True,
            data={
                "ticker": ticker,
                "count": len(records),
                "records": [
                    {
                        "record_type": r.record_type,
                        "data": r.data,
                        "source": r.source,
                        "date": r.date.isoformat(),
                    }
                    for r in records
                ],
            },
            source="AlternativeProvider",
            fetched_at=now,
            is_stale=_is_stale(now),
        )
    except Exception as exc:
        logger.warning("insider failed for %s: %s", ticker, exc)
        return ToolResult(
            tool="insider",
            success=False,
            data={},
            source="AlternativeProvider",
            error=str(exc),
        )


async def macro(indicator: str, registry: DataRegistry) -> ToolResult:
    """Fetch a macroeconomic indicator."""
    try:
        provider: MacroProvider = registry.get(MacroProvider)
        data_point = await provider.get_indicator(indicator)
        now = datetime.now()
        return ToolResult(
            tool="macro",
            success=True,
            data={
                "name": data_point.name,
                "value": data_point.value,
                "date": data_point.date.isoformat(),
                "source": data_point.source,
                "unit": data_point.unit,
            },
            source="MacroProvider",
            fetched_at=now,
            is_stale=_is_stale(now),
        )
    except Exception as exc:
        logger.warning("macro failed for %s: %s", indicator, exc)
        return ToolResult(
            tool="macro", success=False, data={}, source="MacroProvider", error=str(exc)
        )


async def fundamentals(ticker: str, registry: DataRegistry) -> ToolResult:
    """Fetch fundamental financial data for a ticker."""
    try:
        provider: FundamentalProvider = registry.get(FundamentalProvider)
        data = await provider.get_fundamentals(ticker)
        now = datetime.now()
        return ToolResult(
            tool="fundamentals",
            success=True,
            data={
                "ticker": data.ticker,
                "pe_ratio": data.pe_ratio,
                "market_cap": data.market_cap,
                "revenue": data.revenue,
                "earnings": data.earnings,
                "dividend_yield": data.dividend_yield,
            },
            source="FundamentalProvider",
            fetched_at=now,
            is_stale=_is_stale(now),
        )
    except Exception as exc:
        logger.warning("fundamentals failed for %s: %s", ticker, exc)
        return ToolResult(
            tool="fundamentals",
            success=False,
            data={},
            source="FundamentalProvider",
            error=str(exc),
        )


async def cross_market(tickers: list[str], registry: DataRegistry) -> ToolResult:
    """Fetch price data across multiple tickers for cross-market comparison."""
    try:
        provider: PriceProvider = registry.get(PriceProvider)
        end = date.today()
        start = end - timedelta(days=_DEFAULT_LOOKBACK_DAYS)
        result_data: dict[str, object] = {"period_start": start.isoformat(), "tickers": {}}

        for ticker in tickers:
            try:
                bars = await provider.get_ohlcv(ticker, start, end)
                current_price = await provider.get_price(ticker)
                result_data["tickers"][ticker] = {  # type: ignore[index]
                    "current_price": current_price,
                    "bars": len(bars),
                }
            except Exception as inner_exc:
                result_data["tickers"][ticker] = {"error": str(inner_exc)}  # type: ignore[index]

        now = datetime.now()
        return ToolResult(
            tool="cross_market",
            success=True,
            data=result_data,  # type: ignore[arg-type]
            source="PriceProvider",
            fetched_at=now,
            is_stale=_is_stale(now),
        )
    except Exception as exc:
        logger.warning("cross_market failed: %s", exc)
        return ToolResult(
            tool="cross_market",
            success=False,
            data={},
            source="PriceProvider",
            error=str(exc),
        )


async def memory_search(query: str, **kwargs: object) -> ToolResult:
    """Search past analyses stored in session memory.

    This is a placeholder — the full implementation depends on the SessionManager
    and memory system which are built in later layers.
    """
    now = datetime.now()
    return ToolResult(
        tool="memory_search",
        success=True,
        data={"query": query, "results": []},
        source="SessionMemory",
        fetched_at=now,
        is_stale=False,
    )
