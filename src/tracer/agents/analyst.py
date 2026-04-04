"""Analyst agent — cross-market analysis and causal reasoning.

Pipeline steps: Macro Regime Detection (step 2), Cross-Market Discovery (step 3).
"""

from __future__ import annotations

from tracer.agents.base import BaseAgent
from tracer.data.providers import MacroProvider, NewsProvider, PriceProvider
from tracer.llm.providers import Role
from tracer.models import ToolResult


class Analyst(BaseAgent):
    """Deep financial and cross-market analysis.

    Responsibilities:
    - Macro regime detection: risk-on / risk-off / transition.
    - Cross-market discovery: find leading indicators across global markets.
    """

    role = Role.ANALYST

    async def detect_regime(self, indicators: list[str]) -> dict:
        """Determine the current macro regime from a set of indicator names.

        Returns a dict with regime (risk_on/risk_off/transition), reasoning,
        and the indicators that drove the conclusion.
        """
        results: list[ToolResult] = []

        try:
            provider: MacroProvider = self._data.get(MacroProvider)
            for name in indicators:
                try:
                    point = await provider.get_indicator(name)
                    results.append(
                        ToolResult(
                            tool="macro",
                            success=True,
                            data={
                                "name": point.name,
                                "value": point.value,
                                "date": point.date.isoformat(),
                                "source": point.source,
                                "unit": point.unit,
                            },
                            source="MacroProvider",
                        )
                    )
                except Exception:
                    results.append(
                        ToolResult(
                            tool="macro",
                            success=False,
                            data={},
                            source="MacroProvider",
                            error=f"Failed to fetch indicator {name}",
                        )
                    )
        except KeyError:
            pass  # no MacroProvider registered

        data_text = self._format_tool_data(self._successful_results(results))
        response = await self._complete(
            system=(
                "You are a macro analyst. Given macroeconomic indicators, determine "
                "the current market regime. Return a JSON object with keys: "
                "regime (risk_on/risk_off/transition), reasoning (string), "
                "key_drivers (list of indicator names)."
            ),
            user=f"Analyse these macro indicators:\n\n{data_text}",
        )

        import json

        try:
            return json.loads(response.content)
        except (json.JSONDecodeError, ValueError):
            return {"raw": response.content}

    async def discover_cross_market(self, tickers: list[str]) -> dict:
        """Find cross-market signals and leading indicators across tickers.

        Gathers price and news data for all tickers and uses causal reasoning
        to identify information asymmetry.

        Returns a dict with discoveries (list), causal_chains (list), and
        confidence scores.
        """
        results: list[ToolResult] = []
        from datetime import date, timedelta

        end = date.today()
        start = end - timedelta(days=30)

        # Price data
        try:
            price_provider: PriceProvider = self._data.get(PriceProvider)
            for ticker in tickers:
                try:
                    bars = await price_provider.get_ohlcv(ticker, start, end)
                    price = await price_provider.get_price(ticker)
                    results.append(
                        ToolResult(
                            tool="price_event",
                            success=True,
                            data={
                                "ticker": ticker,
                                "current_price": price,
                                "bars": len(bars),
                            },
                            source="PriceProvider",
                        )
                    )
                except Exception:
                    results.append(
                        ToolResult(
                            tool="price_event",
                            success=False,
                            data={},
                            source="PriceProvider",
                            error=f"Price data unavailable for {ticker}",
                        )
                    )
        except KeyError:
            pass

        # News data
        try:
            news_provider: NewsProvider = self._data.get(NewsProvider)
            for ticker in tickers:
                try:
                    articles = await news_provider.get_news(ticker, limit=5)
                    results.append(
                        ToolResult(
                            tool="news",
                            success=True,
                            data={
                                "ticker": ticker,
                                "articles": [
                                    {"title": a.title, "sentiment": a.sentiment} for a in articles
                                ],
                            },
                            source="NewsProvider",
                        )
                    )
                except Exception:
                    pass  # news is supplementary
        except KeyError:
            pass

        data_text = self._format_tool_data(self._successful_results(results))
        response = await self._complete(
            system=(
                "You are a cross-market analyst. Given price and news data across "
                "multiple tickers, identify information asymmetry and leading "
                "indicators. Return a JSON object with keys: discoveries (list of "
                "objects with ticker, signal, reasoning), causal_chains (list of "
                "strings describing A→B relationships)."
            ),
            user=f"Find cross-market signals:\n\n{data_text}",
        )

        import json

        try:
            return json.loads(response.content)
        except (json.JSONDecodeError, ValueError):
            return {"raw": response.content}

    async def run(self, tickers: list[str], **kwargs) -> dict:
        """Default entry point — runs cross-market discovery."""
        return await self.discover_cross_market(tickers, **kwargs)
