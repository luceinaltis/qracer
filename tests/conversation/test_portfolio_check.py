"""Tests for portfolio P&L tracking (QuickPath)."""

from __future__ import annotations

from qracer.conversation.intent import IntentType
from qracer.conversation.quickpath import format_portfolio
from qracer.risk.models import HoldingSnapshot, PortfolioSnapshot


def _snapshot(holdings: list[HoldingSnapshot] | None = None) -> PortfolioSnapshot:
    from datetime import datetime

    if holdings is None:
        holdings = [
            HoldingSnapshot(
                ticker="AAPL",
                shares=100,
                avg_cost=150.0,
                current_price=180.0,
                market_value=18000.0,
                weight_pct=52.94,
                unrealized_pnl=3000.0,
                unrealized_pnl_pct=20.0,
            ),
            HoldingSnapshot(
                ticker="JPM",
                shares=200,
                avg_cost=140.0,
                current_price=80.0,
                market_value=16000.0,
                weight_pct=47.06,
                unrealized_pnl=-12000.0,
                unrealized_pnl_pct=-42.86,
            ),
        ]
    return PortfolioSnapshot(
        holdings=holdings,
        total_value=sum(h.market_value for h in holdings),
        currency="USD",
        as_of=datetime.now(),
    )


class TestFormatPortfolio:
    def test_basic_output(self) -> None:
        text = format_portfolio(_snapshot())
        assert "Portfolio Summary" in text
        assert "AAPL" in text
        assert "JPM" in text
        assert "P&L" in text

    def test_shows_pnl_per_holding(self) -> None:
        text = format_portfolio(_snapshot())
        assert "3,000" in text
        assert "+20.0%" in text
        assert "12,000" in text
        assert "-42.9%" in text

    def test_shows_total(self) -> None:
        text = format_portfolio(_snapshot())
        assert "Total:" in text
        assert "34,000" in text

    def test_empty_holdings(self) -> None:
        text = format_portfolio(_snapshot(holdings=[]))
        assert "No holdings" in text
        assert "portfolio.toml" in text

    def test_positive_total_pnl(self) -> None:
        holdings = [
            HoldingSnapshot(
                ticker="AAPL",
                shares=100,
                avg_cost=150.0,
                current_price=180.0,
                market_value=18000.0,
                weight_pct=100.0,
                unrealized_pnl=3000.0,
                unrealized_pnl_pct=20.0,
            ),
        ]
        text = format_portfolio(_snapshot(holdings=holdings))
        assert "3,000" in text
        assert "+20.0%" in text


class TestPortfolioKeywordDetection:
    async def test_portfolio_keywords(self) -> None:
        from unittest.mock import AsyncMock

        from qracer.conversation.intent import IntentParser
        from qracer.llm.providers import Role
        from qracer.llm.registry import LLMRegistry

        mock = AsyncMock()
        mock.complete.side_effect = RuntimeError("fail")
        registry = LLMRegistry()
        registry.register("mock", mock, [Role.RESEARCHER])
        parser = IntentParser(registry)

        for query in ["How's my portfolio?", "Check my holdings", "Show portfolio"]:
            intent = await parser.parse(query)
            assert intent.intent_type == IntentType.PORTFOLIO_CHECK, f"Failed for: {query}"
