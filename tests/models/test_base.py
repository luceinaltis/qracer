"""Tests for domain models."""

from __future__ import annotations

from datetime import datetime

import pytest

from qracer.models import Region, Report, Sector, Signal, SignalDirection, Stock, ToolResult


class TestStock:
    def test_create_minimal(self) -> None:
        stock = Stock(ticker="AAPL", name="Apple Inc.")
        assert stock.ticker == "AAPL"
        assert stock.name == "Apple Inc."
        assert stock.region == Region.US
        assert stock.currency == "USD"
        assert stock.sector is None
        assert stock.market_cap is None

    def test_create_full(self) -> None:
        stock = Stock(
            ticker="TSMC",
            name="Taiwan Semiconductor",
            sector=Sector.TECHNOLOGY,
            region=Region.ASIA,
            market_cap=800_000_000_000,
            currency="TWD",
        )
        assert stock.sector == Sector.TECHNOLOGY
        assert stock.region == Region.ASIA
        assert stock.market_cap == 800_000_000_000

    def test_frozen(self) -> None:
        stock = Stock(ticker="AAPL", name="Apple Inc.")
        with pytest.raises(AttributeError):
            stock.ticker = "MSFT"  # type: ignore[misc]


class TestSignal:
    def test_create(self) -> None:
        signal = Signal(
            ticker="AAPL",
            direction=SignalDirection.LONG,
            conviction=8.5,
            thesis="Strong earnings momentum",
        )
        assert signal.ticker == "AAPL"
        assert signal.direction == SignalDirection.LONG
        assert signal.conviction == 8.5
        assert signal.evidence == []
        assert signal.risk_factors == []

    def test_conviction_bounds(self) -> None:
        with pytest.raises(ValueError, match="Conviction must be between"):
            Signal(ticker="X", direction=SignalDirection.LONG, conviction=11.0, thesis="t")

        with pytest.raises(ValueError, match="Conviction must be between"):
            Signal(ticker="X", direction=SignalDirection.SHORT, conviction=-1.0, thesis="t")

    def test_conviction_edge_values(self) -> None:
        s0 = Signal(ticker="X", direction=SignalDirection.NEUTRAL, conviction=0.0, thesis="t")
        assert s0.conviction == 0.0
        s10 = Signal(ticker="X", direction=SignalDirection.LONG, conviction=10.0, thesis="t")
        assert s10.conviction == 10.0


class TestReport:
    def test_create(self) -> None:
        report = Report(
            title="AAPL Spike Analysis",
            ticker="AAPL",
            conviction=8.0,
            what_happened="AAPL rose 5% on earnings beat",
            evidence_chain=["Revenue up 12% YoY", "iPhone sales exceeded estimates"],
            adversarial_check=["Could be one-time seasonal effect"],
            verdict="Strong buy signal with high conviction",
        )
        assert report.title == "AAPL Spike Analysis"
        assert len(report.evidence_chain) == 2
        assert report.signals == []


class TestToolResult:
    def test_success(self) -> None:
        result = ToolResult(
            tool="price_event",
            success=True,
            data={"price": 185.0},
            source="Finnhub",
        )
        assert result.success is True
        assert result.error is None
        assert result.is_stale is False
        assert isinstance(result.fetched_at, datetime)

    def test_failure(self) -> None:
        result = ToolResult(
            tool="news",
            success=False,
            data={},
            source="NewsAPI",
            error="Rate limit exceeded",
        )
        assert result.success is False
        assert result.error == "Rate limit exceeded"
