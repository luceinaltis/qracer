"""Tests for pipeline tool wrappers."""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import AsyncMock

from tracer.data.providers import (
    OHLCV,
    AlternativeProvider,
    AlternativeRecord,
    FundamentalData,
    FundamentalProvider,
    MacroIndicator,
    MacroProvider,
    NewsArticle,
    NewsProvider,
    PriceProvider,
)
from tracer.data.registry import DataRegistry
from tracer.tools.pipeline import (
    cross_market,
    fundamentals,
    insider,
    macro,
    memory_search,
    news,
    price_event,
)


def _registry_with(capability: type, mock: object) -> DataRegistry:
    reg = DataRegistry()
    reg.register("mock", mock, [capability])
    return reg


# ---------------------------------------------------------------------------
# price_event
# ---------------------------------------------------------------------------


class TestPriceEvent:
    async def test_success(self) -> None:
        mock = AsyncMock(spec=PriceProvider)
        mock.get_price.return_value = 185.0
        mock.get_ohlcv.return_value = [
            OHLCV(
                date=date(2024, 1, 1), open=180.0, high=186.0,
                low=179.0, close=185.0, volume=1000,
            ),
        ]
        result = await price_event("AAPL", _registry_with(PriceProvider, mock))
        assert result.success is True
        assert result.tool == "price_event"
        assert result.data["ticker"] == "AAPL"
        assert result.data["current_price"] == 185.0
        assert result.error is None

    async def test_failure(self) -> None:
        mock = AsyncMock(spec=PriceProvider)
        mock.get_ohlcv.side_effect = RuntimeError("timeout")
        result = await price_event("AAPL", _registry_with(PriceProvider, mock))
        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# news
# ---------------------------------------------------------------------------


class TestNews:
    async def test_success(self) -> None:
        mock = AsyncMock(spec=NewsProvider)
        mock.get_news.return_value = [
            NewsArticle(
                title="AAPL earnings",
                source="Reuters",
                published_at=datetime(2024, 1, 15),
                url="https://example.com",
                summary="Beat estimates",
                sentiment=0.8,
            ),
        ]
        result = await news("AAPL", _registry_with(NewsProvider, mock))
        assert result.success is True
        assert result.data["count"] == 1
        assert result.data["articles"][0]["title"] == "AAPL earnings"

    async def test_failure(self) -> None:
        mock = AsyncMock(spec=NewsProvider)
        mock.get_news.side_effect = RuntimeError("rate limit")
        result = await news("AAPL", _registry_with(NewsProvider, mock))
        assert result.success is False


# ---------------------------------------------------------------------------
# insider
# ---------------------------------------------------------------------------


class TestInsider:
    async def test_success(self) -> None:
        mock = AsyncMock(spec=AlternativeProvider)
        mock.get_alternative.return_value = [
            AlternativeRecord(
                record_type="insider_trades",
                ticker="AAPL",
                data={"shares": 10000, "direction": "buy"},
                source="SEC",
                date=date(2024, 1, 10),
            ),
        ]
        result = await insider("AAPL", _registry_with(AlternativeProvider, mock))
        assert result.success is True
        assert result.data["count"] == 1

    async def test_failure(self) -> None:
        mock = AsyncMock(spec=AlternativeProvider)
        mock.get_alternative.side_effect = KeyError("no adapter")
        result = await insider("AAPL", _registry_with(AlternativeProvider, mock))
        assert result.success is False


# ---------------------------------------------------------------------------
# macro
# ---------------------------------------------------------------------------


class TestMacro:
    async def test_success(self) -> None:
        mock = AsyncMock(spec=MacroProvider)
        mock.get_indicator.return_value = MacroIndicator(
            name="fed_funds_rate",
            value=5.25,
            date=date(2024, 1, 1),
            source="FRED",
            unit="%",
        )
        result = await macro("fed_funds_rate", _registry_with(MacroProvider, mock))
        assert result.success is True
        assert result.data["value"] == 5.25

    async def test_failure(self) -> None:
        mock = AsyncMock(spec=MacroProvider)
        mock.get_indicator.side_effect = RuntimeError("unavailable")
        result = await macro("cpi", _registry_with(MacroProvider, mock))
        assert result.success is False


# ---------------------------------------------------------------------------
# fundamentals
# ---------------------------------------------------------------------------


class TestFundamentals:
    async def test_success(self) -> None:
        mock = AsyncMock(spec=FundamentalProvider)
        mock.get_fundamentals.return_value = FundamentalData(
            ticker="AAPL",
            pe_ratio=28.5,
            market_cap=3_000_000_000_000,
            revenue=380_000_000_000,
            earnings=95_000_000_000,
            dividend_yield=0.005,
        )
        result = await fundamentals("AAPL", _registry_with(FundamentalProvider, mock))
        assert result.success is True
        assert result.data["pe_ratio"] == 28.5

    async def test_failure(self) -> None:
        mock = AsyncMock(spec=FundamentalProvider)
        mock.get_fundamentals.side_effect = RuntimeError("fail")
        result = await fundamentals("AAPL", _registry_with(FundamentalProvider, mock))
        assert result.success is False


# ---------------------------------------------------------------------------
# cross_market
# ---------------------------------------------------------------------------


class TestCrossMarket:
    async def test_success(self) -> None:
        mock = AsyncMock(spec=PriceProvider)
        mock.get_price.return_value = 100.0
        mock.get_ohlcv.return_value = []
        result = await cross_market(["AAPL", "MSFT"], _registry_with(PriceProvider, mock))
        assert result.success is True
        assert "AAPL" in result.data["tickers"]
        assert "MSFT" in result.data["tickers"]

    async def test_partial_failure(self) -> None:
        """Individual ticker failure should not fail the whole tool."""
        mock = AsyncMock(spec=PriceProvider)
        mock.get_price.side_effect = [100.0, RuntimeError("fail")]
        mock.get_ohlcv.return_value = []
        result = await cross_market(["AAPL", "MSFT"], _registry_with(PriceProvider, mock))
        assert result.success is True
        assert "error" in result.data["tickers"]["MSFT"]

    async def test_registry_failure(self) -> None:
        """If the registry itself fails, the tool returns failure."""
        registry = DataRegistry()  # empty — no providers
        result = await cross_market(["AAPL"], registry)
        assert result.success is False


# ---------------------------------------------------------------------------
# memory_search
# ---------------------------------------------------------------------------


class TestMemorySearch:
    async def test_placeholder(self) -> None:
        result = await memory_search("previous AAPL analysis")
        assert result.success is True
        assert result.tool == "memory_search"
        assert result.data["results"] == []
