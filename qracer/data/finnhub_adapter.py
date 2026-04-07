"""FinnhubAdapter — Fundamental, News, and Alternative data via Finnhub API.

Provides three capabilities:
- FundamentalProvider → PE, market cap, revenue, earnings, dividend yield, sector
- NewsProvider → company news articles with metadata
- AlternativeProvider → insider trading records
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta

from qracer.data.providers import (
    AlternativeRecord,
    FundamentalData,
    NewsArticle,
)

try:
    import finnhub  # pyright: ignore[reportMissingImports]

    _HAS_FINNHUB = True
except ImportError:
    _HAS_FINNHUB = False

# Default lookback for news queries.
_NEWS_LOOKBACK_DAYS = 30


def _fetch_fundamentals(client: object, ticker: str) -> FundamentalData:
    """Synchronous helper — runs in a thread via asyncio.to_thread."""
    profile = client.company_profile2(symbol=ticker)  # type: ignore[union-attr]
    metrics_resp = client.company_basic_financials(ticker, "all")  # type: ignore[union-attr]
    metrics: dict = metrics_resp.get("metric", {}) if metrics_resp else {}

    return FundamentalData(
        ticker=ticker,
        pe_ratio=metrics.get("peNormalizedAnnual"),
        market_cap=profile.get("marketCapitalization"),
        revenue=metrics.get("revenuePerShareTTM"),
        earnings=metrics.get("epsNormalizedAnnual"),
        dividend_yield=metrics.get("dividendYieldIndicatedAnnual"),
        sector=profile.get("finnhubIndustry"),
        fetched_at=datetime.now(),
    )


def _fetch_news(client: object, ticker: str, limit: int) -> list[NewsArticle]:
    """Synchronous helper — runs in a thread via asyncio.to_thread."""
    end = date.today()
    start = end - timedelta(days=_NEWS_LOOKBACK_DAYS)
    articles_raw = client.company_news(  # type: ignore[union-attr]
        ticker, _from=start.isoformat(), to=end.isoformat()
    )
    if not articles_raw:
        return []

    articles: list[NewsArticle] = []
    for item in articles_raw[:limit]:
        published_at = datetime.fromtimestamp(item.get("datetime", 0))
        articles.append(
            NewsArticle(
                title=item.get("headline", ""),
                source=item.get("source", ""),
                published_at=published_at,
                url=item.get("url", ""),
                summary=item.get("summary", ""),
                sentiment=None,
            )
        )
    return articles


def _fetch_insider(client: object, ticker: str) -> list[AlternativeRecord]:
    """Synchronous helper — runs in a thread via asyncio.to_thread."""
    end = date.today()
    start = end - timedelta(days=90)
    resp = client.stock_insider_transactions(  # type: ignore[union-attr]
        ticker, start.isoformat(), end.isoformat()
    )
    records_raw = resp.get("data", []) if resp else []

    records: list[AlternativeRecord] = []
    for item in records_raw:
        tx_date = item.get("transactionDate", "")
        try:
            parsed_date = date.fromisoformat(tx_date) if tx_date else end
        except ValueError:
            parsed_date = end

        records.append(
            AlternativeRecord(
                record_type="insider_trades",
                ticker=ticker,
                data={
                    "name": item.get("name", ""),
                    "share": item.get("share", 0),
                    "change": item.get("change", 0),
                    "transaction_code": item.get("transactionCode", ""),
                    "filing_date": item.get("filingDate", ""),
                },
                source="finnhub",
                date=parsed_date,
            )
        )
    return records


class FinnhubAdapter:
    """Data adapter for Finnhub (Fundamentals, News, Alternative data)."""

    def __init__(self, api_key: str | None = None) -> None:
        if not _HAS_FINNHUB:
            raise ImportError(
                "finnhub-python is not installed. Install it with: uv add finnhub-python"
            )
        if not api_key:
            raise ValueError("FINNHUB_API_KEY is required. Get one at https://finnhub.io/register")
        self._client = finnhub.Client(api_key=api_key)

    async def get_fundamentals(self, ticker: str) -> FundamentalData:
        """Get fundamental financial data for a ticker."""
        return await asyncio.to_thread(_fetch_fundamentals, self._client, ticker)

    async def get_news(self, ticker: str, limit: int = 10) -> list[NewsArticle]:
        """Get recent news articles for a ticker."""
        return await asyncio.to_thread(_fetch_news, self._client, ticker, limit)

    async def get_alternative(self, ticker: str, record_type: str) -> list[AlternativeRecord]:
        """Get alternative data records for a ticker."""
        if record_type == "insider_trades":
            return await asyncio.to_thread(_fetch_insider, self._client, ticker)
        return []
