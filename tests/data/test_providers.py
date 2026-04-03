"""Tests for data provider protocols and data classes."""

from __future__ import annotations

from datetime import date, datetime

from tracer.data.providers import (
    OHLCV,
    AlternativeRecord,
    FundamentalData,
    MacroIndicator,
    NewsArticle,
    PriceProvider,
)


class TestOHLCV:
    def test_create(self) -> None:
        bar = OHLCV(
            date=date(2026, 1, 15),
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=1_000_000,
        )
        assert bar.close == 153.0
        assert bar.volume == 1_000_000

    def test_frozen(self) -> None:
        bar = OHLCV(date=date.today(), open=1, high=2, low=0.5, close=1.5, volume=100)
        try:
            bar.close = 999  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass


class TestFundamentalData:
    def test_defaults(self) -> None:
        data = FundamentalData(ticker="AAPL")
        assert data.pe_ratio is None
        assert data.market_cap is None


class TestMacroIndicator:
    def test_create(self) -> None:
        ind = MacroIndicator(name="CPI", value=3.2, date=date(2026, 3, 1), source="FRED")
        assert ind.name == "CPI"
        assert ind.unit == ""


class TestNewsArticle:
    def test_create(self) -> None:
        article = NewsArticle(
            title="AAPL beats earnings",
            source="Reuters",
            published_at=datetime(2026, 1, 15, 10, 0),
            url="https://example.com/article",
        )
        assert article.sentiment is None
        assert article.summary == ""


class TestAlternativeRecord:
    def test_create(self) -> None:
        record = AlternativeRecord(
            record_type="insider_trade",
            ticker="AAPL",
            data={"shares": 10000, "direction": "buy"},
            source="SEC EDGAR",
            date=date(2026, 1, 10),
        )
        assert record.record_type == "insider_trade"


class TestProtocolConformance:
    def test_price_provider_protocol(self) -> None:
        """Verify that a class with the right methods satisfies PriceProvider."""

        class MyAdapter:
            async def get_price(self, ticker: str) -> float:
                return 0.0

            async def get_ohlcv(self, ticker: str, start: date, end: date) -> list[OHLCV]:
                return []

        assert isinstance(MyAdapter(), PriceProvider)
