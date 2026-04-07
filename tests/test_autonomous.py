"""Tests for the AutonomousMonitor."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from qracer.autonomous import (
    AutonomousAlert,
    AutonomousMonitor,
    Severity,
    TriggerType,
    is_market_open,
)
from qracer.data.providers import NewsArticle


# ---------------------------------------------------------------------------
# is_market_open
# ---------------------------------------------------------------------------


class TestIsMarketOpen:
    def test_open_during_trading_hours(self) -> None:
        # Wednesday 11:00 AM ET
        et = ZoneInfo("America/New_York")
        dt = datetime(2026, 4, 8, 11, 0, tzinfo=et)
        assert is_market_open(dt) is True

    def test_closed_before_open(self) -> None:
        et = ZoneInfo("America/New_York")
        dt = datetime(2026, 4, 8, 9, 0, tzinfo=et)
        assert is_market_open(dt) is False

    def test_closed_after_close(self) -> None:
        et = ZoneInfo("America/New_York")
        dt = datetime(2026, 4, 8, 16, 0, tzinfo=et)
        assert is_market_open(dt) is False

    def test_open_at_exactly_930(self) -> None:
        et = ZoneInfo("America/New_York")
        dt = datetime(2026, 4, 8, 9, 30, tzinfo=et)
        assert is_market_open(dt) is True

    def test_closed_on_saturday(self) -> None:
        et = ZoneInfo("America/New_York")
        dt = datetime(2026, 4, 11, 12, 0, tzinfo=et)  # Saturday
        assert is_market_open(dt) is False

    def test_closed_on_sunday(self) -> None:
        et = ZoneInfo("America/New_York")
        dt = datetime(2026, 4, 12, 12, 0, tzinfo=et)  # Sunday
        assert is_market_open(dt) is False


# ---------------------------------------------------------------------------
# AutonomousAlert dataclass
# ---------------------------------------------------------------------------


class TestAutonomousAlert:
    def test_creation(self) -> None:
        alert = AutonomousAlert(
            ticker="AAPL",
            trigger=TriggerType.PRICE_MOVE,
            summary="AAPL moved up 3.0%",
            severity=Severity.MEDIUM,
        )
        assert alert.ticker == "AAPL"
        assert alert.trigger == TriggerType.PRICE_MOVE
        assert alert.severity == Severity.MEDIUM
        assert alert.data == {}
        assert alert.timestamp is not None


# ---------------------------------------------------------------------------
# AutonomousMonitor
# ---------------------------------------------------------------------------


def _make_watchlist(tickers: list[str]) -> MagicMock:
    wl = MagicMock()
    wl.tickers = tickers
    return wl


def _make_registry(price: float | None = None, news: list | None = None) -> MagicMock:
    registry = MagicMock()

    def get_side_effect(capability):
        from qracer.data.providers import NewsProvider, PriceProvider

        if capability is PriceProvider:
            if price is None:
                raise KeyError("No PriceProvider")
            provider = MagicMock()
            provider.get_price = AsyncMock(return_value=price)
            return provider
        if capability is NewsProvider:
            if news is None:
                raise KeyError("No NewsProvider")
            provider = MagicMock()
            provider.get_news = AsyncMock(return_value=news)
            return provider
        raise KeyError(f"No {capability}")

    registry.get = MagicMock(side_effect=get_side_effect)
    return registry


class TestAutonomousMonitor:
    def test_should_check_initially_true(self) -> None:
        monitor = AutonomousMonitor(
            _make_watchlist([]),
            _make_registry(),
            check_interval=60.0,
        )
        assert monitor.should_check() is True

    def test_should_check_respects_interval(self) -> None:
        monitor = AutonomousMonitor(
            _make_watchlist([]),
            _make_registry(),
            check_interval=60.0,
        )
        monitor._last_check = time.monotonic()
        assert monitor.should_check() is False

    def test_should_check_after_interval(self) -> None:
        monitor = AutonomousMonitor(
            _make_watchlist([]),
            _make_registry(),
            check_interval=1.0,
        )
        monitor._last_check = time.monotonic() - 2.0
        assert monitor.should_check() is True

    @patch("qracer.autonomous.is_market_open", return_value=False)
    async def test_check_returns_empty_when_market_closed(self, _mock: MagicMock) -> None:
        monitor = AutonomousMonitor(
            _make_watchlist(["AAPL"]),
            _make_registry(price=150.0),
        )
        result = await monitor.check()
        assert result == []

    @patch("qracer.autonomous.is_market_open", return_value=True)
    async def test_check_returns_empty_with_no_watchlist(self, _mock: MagicMock) -> None:
        monitor = AutonomousMonitor(
            _make_watchlist([]),
            _make_registry(price=150.0),
        )
        result = await monitor.check()
        assert result == []

    @patch("qracer.autonomous.is_market_open", return_value=True)
    async def test_check_no_alert_on_first_price(self, _mock: MagicMock) -> None:
        """First price fetch establishes baseline — no alert expected."""
        monitor = AutonomousMonitor(
            _make_watchlist(["AAPL"]),
            _make_registry(price=150.0),
        )
        result = await monitor.check()
        assert result == []
        assert monitor._baseline_prices["AAPL"] == 150.0

    @patch("qracer.autonomous.is_market_open", return_value=True)
    async def test_check_no_alert_below_threshold(self, _mock: MagicMock) -> None:
        """Small move below threshold should not trigger."""
        monitor = AutonomousMonitor(
            _make_watchlist(["AAPL"]),
            _make_registry(price=151.0),
            price_move_threshold_pct=5.0,
        )
        monitor._baseline_prices["AAPL"] = 150.0
        result = await monitor.check()
        assert result == []

    @patch("qracer.autonomous.is_market_open", return_value=True)
    async def test_check_alerts_on_price_move_up(self, _mock: MagicMock) -> None:
        monitor = AutonomousMonitor(
            _make_watchlist(["AAPL"]),
            _make_registry(price=155.0),
            price_move_threshold_pct=2.0,
        )
        monitor._baseline_prices["AAPL"] = 150.0
        result = await monitor.check()
        assert len(result) == 1
        assert result[0].ticker == "AAPL"
        assert result[0].trigger == TriggerType.PRICE_MOVE
        assert result[0].severity == Severity.MEDIUM
        assert "up" in result[0].summary

    @patch("qracer.autonomous.is_market_open", return_value=True)
    async def test_check_alerts_on_price_move_down(self, _mock: MagicMock) -> None:
        monitor = AutonomousMonitor(
            _make_watchlist(["AAPL"]),
            _make_registry(price=145.0),
            price_move_threshold_pct=2.0,
        )
        monitor._baseline_prices["AAPL"] = 150.0
        result = await monitor.check()
        assert len(result) == 1
        assert "down" in result[0].summary

    @patch("qracer.autonomous.is_market_open", return_value=True)
    async def test_high_severity_on_large_move(self, _mock: MagicMock) -> None:
        monitor = AutonomousMonitor(
            _make_watchlist(["AAPL"]),
            _make_registry(price=160.0),
            price_move_threshold_pct=2.0,
        )
        monitor._baseline_prices["AAPL"] = 150.0
        result = await monitor.check()
        assert len(result) == 1
        assert result[0].severity == Severity.HIGH

    @patch("qracer.autonomous.is_market_open", return_value=True)
    async def test_cooldown_prevents_repeated_alerts(self, _mock: MagicMock) -> None:
        monitor = AutonomousMonitor(
            _make_watchlist(["AAPL"]),
            _make_registry(price=155.0),
            price_move_threshold_pct=2.0,
            alert_cooldown_minutes=30,
        )
        monitor._baseline_prices["AAPL"] = 150.0

        # First check triggers alert
        result = await monitor.check()
        assert len(result) == 1

        # Second check should be suppressed (cooldown)
        monitor._baseline_prices["AAPL"] = 150.0  # Reset baseline
        result = await monitor.check()
        assert result == []

    @patch("qracer.autonomous.is_market_open", return_value=True)
    async def test_cooldown_expires(self, _mock: MagicMock) -> None:
        monitor = AutonomousMonitor(
            _make_watchlist(["AAPL"]),
            _make_registry(price=155.0),
            price_move_threshold_pct=2.0,
            alert_cooldown_minutes=30,
        )
        monitor._baseline_prices["AAPL"] = 150.0

        # First alert
        await monitor.check()

        # Simulate cooldown expiry
        monitor._last_alert_time["AAPL"] = time.monotonic() - 1801
        monitor._baseline_prices["AAPL"] = 150.0

        result = await monitor.check()
        assert len(result) == 1

    @patch("qracer.autonomous.is_market_open", return_value=True)
    async def test_breaking_news_alert(self, _mock: MagicMock) -> None:
        article = NewsArticle(
            title="AAPL surges on AI announcement",
            source="Reuters",
            published_at=datetime.now(tz=timezone.utc),
            url="https://example.com/news",
            summary="Apple announces...",
            sentiment=0.9,
        )
        registry = _make_registry(news=[article])
        monitor = AutonomousMonitor(
            _make_watchlist(["AAPL"]),
            registry,
        )
        result = await monitor.check()
        assert len(result) == 1
        assert result[0].trigger == TriggerType.BREAKING_NEWS
        assert "AAPL surges" in result[0].summary

    @patch("qracer.autonomous.is_market_open", return_value=True)
    async def test_low_sentiment_news_ignored(self, _mock: MagicMock) -> None:
        article = NewsArticle(
            title="AAPL routine update",
            source="Reuters",
            published_at=datetime.now(tz=timezone.utc),
            url="https://example.com/news",
            summary="Nothing special",
            sentiment=0.3,
        )
        registry = _make_registry(news=[article])
        monitor = AutonomousMonitor(
            _make_watchlist(["AAPL"]),
            registry,
        )
        result = await monitor.check()
        assert result == []

    @patch("qracer.autonomous.is_market_open", return_value=True)
    async def test_no_price_provider_graceful(self, _mock: MagicMock) -> None:
        """Should handle missing PriceProvider gracefully."""
        registry = _make_registry(price=None, news=None)
        monitor = AutonomousMonitor(
            _make_watchlist(["AAPL"]),
            registry,
        )
        result = await monitor.check()
        assert result == []


# ---------------------------------------------------------------------------
# Server integration (tick includes autonomous)
# ---------------------------------------------------------------------------


class TestServerAutonomousIntegration:
    async def test_tick_calls_autonomous_monitor(self) -> None:
        from qracer.server import Server

        alert_monitor = MagicMock()
        alert_monitor.should_check.return_value = False
        task_executor = MagicMock()
        task_executor.should_check.return_value = False

        auto_monitor = MagicMock()
        auto_monitor.should_check.return_value = True
        auto_alert = AutonomousAlert(
            ticker="AAPL",
            trigger=TriggerType.PRICE_MOVE,
            summary="AAPL moved up 3.0%",
            severity=Severity.MEDIUM,
        )
        auto_monitor.check = AsyncMock(return_value=[auto_alert])

        notifications = MagicMock()
        notifications.channels = ["telegram"]
        notifications.notify = AsyncMock(return_value={"telegram": True})

        server = Server(
            alert_monitor,
            task_executor,
            notifications,
            autonomous_monitor=auto_monitor,
        )
        await server._tick()

        auto_monitor.should_check.assert_called_once()
        auto_monitor.check.assert_called_once()
        notifications.notify.assert_called_once()

    async def test_tick_skips_autonomous_when_none(self) -> None:
        from qracer.server import Server

        alert_monitor = MagicMock()
        alert_monitor.should_check.return_value = False
        task_executor = MagicMock()
        task_executor.should_check.return_value = False

        server = Server(alert_monitor, task_executor, autonomous_monitor=None)
        await server._tick()  # Should not raise

    async def test_tick_handles_autonomous_error(self) -> None:
        from qracer.server import Server

        alert_monitor = MagicMock()
        alert_monitor.should_check.return_value = False
        task_executor = MagicMock()
        task_executor.should_check.return_value = False

        auto_monitor = MagicMock()
        auto_monitor.should_check.return_value = True
        auto_monitor.check = AsyncMock(side_effect=RuntimeError("boom"))

        server = Server(
            alert_monitor, task_executor, autonomous_monitor=auto_monitor
        )
        await server._tick()  # Should not raise
