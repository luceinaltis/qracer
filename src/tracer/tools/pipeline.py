"""Pipeline tool wrappers — thin async functions that fetch data and return ToolResult.

Each tool wraps a DataRegistry capability lookup, calls the provider, and
normalises the output into the uniform ToolResult contract consumed by the
AnalysisLoop.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from tracer.config.models import PortfolioConfig
from tracer.data.providers import (
    AlternativeProvider,
    FundamentalProvider,
    MacroProvider,
    NewsProvider,
    PriceProvider,
)
from tracer.data.registry import DataRegistry
from tracer.llm.providers import CompletionRequest, Message, Role
from tracer.llm.registry import LLMRegistry
from tracer.models import ToolResult, TradeThesis

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


_TRADE_THESIS_SYSTEM = (
    "You are a senior investment strategist. Given the analysis results for a ticker, "
    "generate a structured trade thesis.\n\n"
    "Return ONLY valid JSON with these exact keys:\n"
    '- "entry_zone": [low, high] (price range for entry)\n'
    '- "target_price": float\n'
    '- "stop_loss": float\n'
    '- "catalyst": string (what drives the move)\n'
    '- "catalyst_date": string or null (expected timing, e.g. "Q2 2026")\n'
    '- "conviction": integer 1-10\n'
    '- "summary": string (one-paragraph thesis)\n\n'
    "Base entry zone on current price data. Ensure stop_loss < entry_zone[0] < "
    "entry_zone[1] < target_price for long trades."
)


async def trade_thesis(
    ticker: str, analysis_results: list[ToolResult], llm_registry: LLMRegistry
) -> ToolResult:
    """Generate a structured trade thesis from prior analysis results (step 7)."""
    import json

    evidence_parts: list[str] = []
    for r in analysis_results:
        if r.success:
            evidence_parts.append(f"[{r.tool}] source={r.source}\n{json.dumps(r.data, indent=2)}")

    evidence = "\n\n".join(evidence_parts) if evidence_parts else "(no data)"
    user_msg = f"Ticker: {ticker}\n\nAnalysis results:\n{evidence}"

    try:
        provider = llm_registry.get(Role.STRATEGIST)
        response = await provider.complete(
            CompletionRequest(
                messages=[
                    Message(role="system", content=_TRADE_THESIS_SYSTEM),
                    Message(role="user", content=user_msg),
                ],
                max_tokens=1024,
                temperature=0.2,
            )
        )

        parsed = json.loads(response.content)

        entry_zone = (float(parsed["entry_zone"][0]), float(parsed["entry_zone"][1]))
        target_price = float(parsed["target_price"])
        stop_loss = float(parsed["stop_loss"])
        entry_mid = (entry_zone[0] + entry_zone[1]) / 2
        denominator = entry_mid - stop_loss
        risk_reward_ratio = (target_price - entry_mid) / denominator if denominator > 0 else 0.0

        thesis = TradeThesis(
            ticker=ticker,
            entry_zone=entry_zone,
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=round(risk_reward_ratio, 2),
            catalyst=parsed["catalyst"],
            catalyst_date=parsed.get("catalyst_date"),
            conviction=int(parsed["conviction"]),
            summary=parsed["summary"],
        )

        now = datetime.now()
        return ToolResult(
            tool="trade_thesis",
            success=True,
            data={
                "thesis": {
                    "ticker": thesis.ticker,
                    "entry_zone": list(thesis.entry_zone),
                    "target_price": thesis.target_price,
                    "stop_loss": thesis.stop_loss,
                    "risk_reward_ratio": thesis.risk_reward_ratio,
                    "catalyst": thesis.catalyst,
                    "catalyst_date": thesis.catalyst_date,
                    "conviction": thesis.conviction,
                    "summary": thesis.summary,
                }
            },
            source="LLM/strategist",
            fetched_at=now,
            is_stale=False,
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.warning("trade_thesis parsing failed for %s: %s", ticker, exc)
        return ToolResult(
            tool="trade_thesis",
            success=False,
            data={},
            source="LLM/strategist",
            error=f"Failed to parse LLM response: {exc}",
        )
    except Exception as exc:
        logger.warning("trade_thesis failed for %s: %s", ticker, exc)
        return ToolResult(
            tool="trade_thesis",
            success=False,
            data={},
            source="LLM/strategist",
            error=str(exc),
        )


async def risk_check(
    ticker: str,
    thesis: TradeThesis,
    registry: DataRegistry,
    config: PortfolioConfig,
) -> ToolResult:
    """Run risk check on a proposed trade (step 8).

    Builds a portfolio snapshot from current prices, sizes the proposed
    position using conviction from the thesis, and checks portfolio limits.
    """
    from tracer.data.providers import PriceProvider
    from tracer.risk.calculator import RiskCalculator

    try:
        calculator = RiskCalculator(config)

        # Gather current prices for all holdings.
        price_provider: PriceProvider = registry.get(PriceProvider)
        prices: dict[str, float] = {}
        for holding in config.holdings:
            try:
                prices[holding.ticker] = await price_provider.get_price(holding.ticker)
            except Exception as exc:
                logger.warning("Could not fetch price for %s: %s", holding.ticker, exc)

        snapshot = calculator.build_snapshot(prices)
        exposure = calculator.build_exposure(snapshot)

        # Size the proposed position.
        allocation_pct = calculator.size_position(ticker, thesis.conviction, snapshot)

        # Check limits (including the hypothetical new position).
        breached = calculator.check_limits(snapshot, exposure)

        now = datetime.now()
        return ToolResult(
            tool="risk_check",
            success=True,
            data={
                "ticker": ticker,
                "conviction": thesis.conviction,
                "allocation_pct": allocation_pct,
                "portfolio_value": snapshot.total_value,
                "holdings_count": len(snapshot.holdings),
                "top_sector": exposure.top_sector,
                "top_sector_pct": exposure.top_sector_pct,
                "limits_breached": breached,
                "sized_recommendation": (
                    f"Allocate {allocation_pct:.2f}% of portfolio to {ticker}"
                    if not breached
                    else f"Allocate {allocation_pct:.2f}% of portfolio to {ticker} "
                    f"(WARNING: {len(breached)} limit(s) breached)"
                ),
            },
            source="RiskCalculator",
            fetched_at=now,
            is_stale=False,
        )
    except Exception as exc:
        logger.warning("risk_check failed for %s: %s", ticker, exc)
        return ToolResult(
            tool="risk_check",
            success=False,
            data={},
            source="RiskCalculator",
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
