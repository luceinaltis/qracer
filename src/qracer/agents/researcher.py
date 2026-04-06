"""Researcher agent — data gathering and universe screening.

Pipeline steps: Universe Screening (step 1), Consensus Mapping (step 4).
"""

from __future__ import annotations

from qracer.agents.base import BaseAgent
from qracer.data.providers import (
    FundamentalProvider,
    NewsProvider,
)
from qracer.llm.providers import Role
from qracer.models import Stock, ToolResult


class Researcher(BaseAgent):
    """Gathers and summarises market data.

    Responsibilities:
    - Universe screening: filter markets by region, sector, liquidity.
    - Consensus mapping: collect analyst ratings, sentiment, positioning.
    """

    role = Role.RESEARCHER

    async def screen_universe(
        self,
        tickers: list[str],
        *,
        min_market_cap: float | None = None,
    ) -> list[Stock]:
        """Screen tickers and return those that pass fundamental filters.

        Uses PriceProvider + FundamentalProvider to gather data, then asks
        the researcher LLM to select the most actionable subset.
        """
        results: list[ToolResult] = []

        # Gather fundamentals for each ticker
        try:
            provider: FundamentalProvider = self._data.get(FundamentalProvider)
            for ticker in tickers:
                try:
                    data = await provider.get_fundamentals(ticker)
                    results.append(
                        ToolResult(
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
                        )
                    )
                except Exception:
                    results.append(
                        ToolResult(
                            tool="fundamentals",
                            success=False,
                            data={},
                            source="FundamentalProvider",
                            error=f"Failed to fetch fundamentals for {ticker}",
                        )
                    )
        except KeyError:
            pass  # no FundamentalProvider registered

        successful = self._successful_results(results)
        if not successful:
            return []

        # Apply hard market-cap filter before LLM call
        filtered = successful
        if min_market_cap is not None:
            filtered = [r for r in successful if (r.data.get("market_cap") or 0) >= min_market_cap]

        if not filtered:
            return []

        # Ask LLM to rank and select
        data_text = self._format_tool_data(filtered)
        selected = await self._complete_json(
            system=(
                "You are a market researcher. Given fundamental data for a set of "
                "tickers, return a JSON array of tickers that are most actionable "
                "for further analysis. Consider liquidity, valuation, and earnings "
                "quality. Return ONLY a JSON array of ticker strings."
            ),
            user=f"Screen the following tickers:\n\n{data_text}",
            fallback=[r.data["ticker"] for r in filtered if "ticker" in r.data],
        )

        if not isinstance(selected, list):
            selected = [r.data["ticker"] for r in filtered if "ticker" in r.data]

        return [Stock(ticker=t, name=t) for t in selected if isinstance(t, str)]

    async def map_consensus(self, ticker: str) -> dict:
        """Build a consensus view for a ticker using news and alternative data.

        Returns a dict with sentiment summary, key themes, and data sources used.
        """
        results: list[ToolResult] = []

        # News
        try:
            news_provider: NewsProvider = self._data.get(NewsProvider)
            articles = await news_provider.get_news(ticker, limit=10)
            results.append(
                ToolResult(
                    tool="news",
                    success=True,
                    data={
                        "ticker": ticker,
                        "count": len(articles),
                        "articles": [
                            {
                                "title": a.title,
                                "source": a.source,
                                "summary": a.summary,
                                "sentiment": a.sentiment,
                            }
                            for a in articles
                        ],
                    },
                    source="NewsProvider",
                )
            )
        except KeyError:
            results.append(
                ToolResult(
                    tool="news",
                    success=False,
                    data={},
                    source="NewsProvider",
                    error=f"News unavailable for {ticker}",
                )
            )

        # Fundamentals
        try:
            fund_provider: FundamentalProvider = self._data.get(FundamentalProvider)
            fundamentals = await fund_provider.get_fundamentals(ticker)
            results.append(
                ToolResult(
                    tool="fundamentals",
                    success=True,
                    data={
                        "ticker": fundamentals.ticker,
                        "pe_ratio": fundamentals.pe_ratio,
                        "market_cap": fundamentals.market_cap,
                    },
                    source="FundamentalProvider",
                )
            )
        except KeyError:
            results.append(
                ToolResult(
                    tool="fundamentals",
                    success=False,
                    data={},
                    source="FundamentalProvider",
                    error=f"Fundamentals unavailable for {ticker}",
                )
            )

        data_text = self._format_tool_data(self._successful_results(results))
        return await self._complete_json(
            system=(
                "You are a market researcher. Summarise the consensus view for "
                "this ticker. Return a JSON object with keys: sentiment (bullish/"
                "bearish/neutral), themes (list of strings), and confidence (0-10)."
            ),
            user=f"Build consensus view for {ticker}:\n\n{data_text}",
        )

    async def run(self, tickers: list[str], **kwargs) -> list[Stock]:
        """Default entry point — runs universe screening."""
        return await self.screen_universe(tickers, **kwargs)
