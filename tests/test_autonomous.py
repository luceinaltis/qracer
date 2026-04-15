"""Tests for AutonomousMonitor."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from qracer.autonomous import (
    AutonomousAlert,
    AutonomousAlertStore,
    AutonomousMonitor,
    Severity,
    TriggerType,
    _severity_for_pct,
    is_market_hours,
)
from qracer.data.providers import NewsArticle, NewsProvider, PriceProvider
from qracer.data.registry import DataRegistry
from qracer.watchlist import Watchlist

US_EASTERN = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakePriceProvider:
    """Returns configurable prices for tickers."""

    def __init__(self, prices: dict[str, float]) -> None:
        self._prices = prices

    async def get_price(self, ticker: str) -> float:
        if ticker not in self._prices:
            raise KeyError(f"No price for {ticker}")
        return self._prices[ticker]

    async def get_ohlcv(self, ticker, start, end):
        return []


class FakeNewsProvider:
    """Returns configurable news articles."""

    def __init__(self, articles: dict[str, list[NewsArticle]]) -> None:
        self._articles = articles

    async def get_news(self, ticker: str, limit: int = 10) -> list[NewsArticle]:
        return self._articles.get(ticker, [])[:limit]


class FailingPriceProvider:
    async def get_price(self, ticker: str) -> float:
        raise ConnectionError("API down")

    async def get_ohlcv(self, ticker, start, end):
        return []


class FailingNewsProvider:
    async def get_news(self, ticker: str, limit: int = 10) -> list[NewsArticle]:
        raise ConnectionError("API down")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry(
    price_provider: object | None = None,
    news_provider: object | None = None,
) -> DataRegistry:
    registry = DataRegistry()
    if price_provider is not None:
        registry.register("fake_price", price_provider, [PriceProvider])
    if news_provider is not None:
        registry.register("fake_news", news_provider, [NewsProvider])
    return registry


def _make_article(
    ticker: str,
    title: str = "Breaking news",
    sentiment: float | None = None,
) -> NewsArticle:
    return NewsArticle(
        title=title,
        source="TestSource",
        published_at=datetime.now(timezone.utc),
        url="https://example.com/article",
        summary="Test summary",
        sentiment=sentiment,
    )


@pytest.fixture
def watchlist(tmp_path):
    wl = Watchlist(tmp_path / "watchlist.json")
    wl.add("AAPL")
    wl.add("TSLA")
    return wl


@pytest.fixture
def empty_watchlist(tmp_path):
    return Watchlist(tmp_path / "watchlist_empty.json")


# ---------------------------------------------------------------------------
# is_market_hours
# ---------------------------------------------------------------------------


class TestIsMarketHours:
    def test_during_market_hours(self) -> None:
        # Wednesday 11:00 ET
        dt = datetime(2026, 4, 8, 11, 0, tzinfo=US_EASTERN)
        assert is_market_hours(dt) is True

    def test_before_market_open(self) -> None:
        # Wednesday 09:00 ET
        dt = datetime(2026, 4, 8, 9, 0, tzinfo=US_EASTERN)
        assert is_market_hours(dt) is False

    def test_after_market_close(self) -> None:
        # Wednesday 16:30 ET
        dt = datetime(2026, 4, 8, 16, 30, tzinfo=US_EASTERN)
        assert is_market_hours(dt) is False

    def test_at_market_open(self) -> None:
        # Exactly 09:30 ET
        dt = datetime(2026, 4, 8, 9, 30, tzinfo=US_EASTERN)
        assert is_market_hours(dt) is True

    def test_at_market_close(self) -> None:
        # Exactly 16:00 ET
        dt = datetime(2026, 4, 8, 16, 0, tzinfo=US_EASTERN)
        assert is_market_hours(dt) is True

    def test_weekend_saturday(self) -> None:
        # Saturday 12:00 ET
        dt = datetime(2026, 4, 11, 12, 0, tzinfo=US_EASTERN)
        assert is_market_hours(dt) is False

    def test_weekend_sunday(self) -> None:
        # Sunday 12:00 ET
        dt = datetime(2026, 4, 12, 12, 0, tzinfo=US_EASTERN)
        assert is_market_hours(dt) is False

    def test_utc_time_converted(self) -> None:
        # 15:00 UTC on a Wednesday = 11:00 ET (during market hours)
        dt = datetime(2026, 4, 8, 15, 0, tzinfo=timezone.utc)
        assert is_market_hours(dt) is True


# ---------------------------------------------------------------------------
# _severity_for_pct
# ---------------------------------------------------------------------------


class TestSeverityForPct:
    def test_info_level(self) -> None:
        assert _severity_for_pct(2.0) == Severity.INFO

    def test_warning_level(self) -> None:
        assert _severity_for_pct(3.5) == Severity.WARNING

    def test_critical_level(self) -> None:
        assert _severity_for_pct(5.0) == Severity.CRITICAL

    def test_negative_pct(self) -> None:
        assert _severity_for_pct(-4.0) == Severity.WARNING

    def test_large_negative(self) -> None:
        assert _severity_for_pct(-7.0) == Severity.CRITICAL


# ---------------------------------------------------------------------------
# AutonomousMonitor.should_check
# ---------------------------------------------------------------------------


class TestShouldCheck:
    def test_first_check_during_market_hours(self, watchlist) -> None:
        registry = _make_registry(FakePriceProvider({}))
        monitor = AutonomousMonitor(watchlist, registry)
        # Patch is_market_hours to return True
        with patch("qracer.autonomous.is_market_hours", return_value=True):
            assert monitor.should_check() is True

    def test_not_during_market_hours(self, watchlist) -> None:
        registry = _make_registry(FakePriceProvider({}))
        monitor = AutonomousMonitor(watchlist, registry)
        with patch("qracer.autonomous.is_market_hours", return_value=False):
            assert monitor.should_check() is False

    async def test_false_after_recent_check(self, watchlist) -> None:
        registry = _make_registry(FakePriceProvider({}))
        monitor = AutonomousMonitor(watchlist, registry, check_interval=60)
        with patch("qracer.autonomous.is_market_hours", return_value=True):
            await monitor.check()
            assert monitor.should_check() is False


# ---------------------------------------------------------------------------
# AutonomousMonitor.check — price moves
# ---------------------------------------------------------------------------


class TestPriceMoves:
    async def test_no_alert_on_first_check(self, watchlist) -> None:
        """First check establishes baseline — no alerts."""
        provider = FakePriceProvider({"AAPL": 200.0, "TSLA": 300.0})
        registry = _make_registry(price_provider=provider)
        monitor = AutonomousMonitor(watchlist, registry, check_interval=0)

        alerts = await monitor.check()
        assert len(alerts) == 0

    async def test_price_move_triggers_alert(self, watchlist) -> None:
        provider = FakePriceProvider({"AAPL": 200.0, "TSLA": 300.0})
        registry = _make_registry(price_provider=provider)
        monitor = AutonomousMonitor(watchlist, registry, check_interval=0, cooldown_minutes=0)

        # First check — baseline
        await monitor.check()

        # Simulate price jump
        provider._prices["AAPL"] = 210.0  # +5%
        alerts = await monitor.check()

        price_alerts = [a for a in alerts if a.trigger_type == TriggerType.PRICE_MOVE]
        assert len(price_alerts) == 1
        assert price_alerts[0].ticker == "AAPL"
        assert "up" in price_alerts[0].summary
        assert price_alerts[0].severity == Severity.CRITICAL

    async def test_small_move_no_alert(self, watchlist) -> None:
        provider = FakePriceProvider({"AAPL": 200.0, "TSLA": 300.0})
        registry = _make_registry(price_provider=provider)
        monitor = AutonomousMonitor(watchlist, registry, check_interval=0, cooldown_minutes=0)

        await monitor.check()
        provider._prices["AAPL"] = 201.0  # +0.5%
        alerts = await monitor.check()

        price_alerts = [a for a in alerts if a.trigger_type == TriggerType.PRICE_MOVE]
        assert len(price_alerts) == 0

    async def test_price_drop_alert(self, watchlist) -> None:
        provider = FakePriceProvider({"AAPL": 200.0, "TSLA": 300.0})
        registry = _make_registry(price_provider=provider)
        monitor = AutonomousMonitor(watchlist, registry, check_interval=0, cooldown_minutes=0)

        await monitor.check()
        provider._prices["TSLA"] = 285.0  # -5%
        alerts = await monitor.check()

        price_alerts = [a for a in alerts if a.trigger_type == TriggerType.PRICE_MOVE]
        assert len(price_alerts) == 1
        assert price_alerts[0].ticker == "TSLA"
        assert "down" in price_alerts[0].summary

    async def test_custom_threshold(self, watchlist) -> None:
        provider = FakePriceProvider({"AAPL": 200.0, "TSLA": 300.0})
        registry = _make_registry(price_provider=provider)
        monitor = AutonomousMonitor(
            watchlist, registry, check_interval=0, cooldown_minutes=0, price_threshold_pct=5.0
        )

        await monitor.check()
        provider._prices["AAPL"] = 206.0  # +3% — below 5% threshold
        alerts = await monitor.check()

        price_alerts = [a for a in alerts if a.trigger_type == TriggerType.PRICE_MOVE]
        assert len(price_alerts) == 0

    async def test_no_provider_no_crash(self, watchlist) -> None:
        registry = _make_registry()  # no price provider
        monitor = AutonomousMonitor(watchlist, registry, check_interval=0)

        alerts = await monitor.check()
        assert len(alerts) == 0

    async def test_price_fetch_failure(self, watchlist) -> None:
        provider = FailingPriceProvider()
        registry = _make_registry(price_provider=provider)
        monitor = AutonomousMonitor(watchlist, registry, check_interval=0)

        alerts = await monitor.check()
        assert len(alerts) == 0

    async def test_baseline_resets_after_alert(self, watchlist) -> None:
        """After an alert, the baseline is updated so the same move doesn't re-trigger."""
        provider = FakePriceProvider({"AAPL": 200.0, "TSLA": 300.0})
        registry = _make_registry(price_provider=provider)
        monitor = AutonomousMonitor(watchlist, registry, check_interval=0, cooldown_minutes=0)

        await monitor.check()
        provider._prices["AAPL"] = 210.0  # +5%
        alerts1 = await monitor.check()
        assert len([a for a in alerts1 if a.trigger_type == TriggerType.PRICE_MOVE]) == 1

        # Same price — no new alert
        alerts2 = await monitor.check()
        price_alerts = [a for a in alerts2 if a.trigger_type == TriggerType.PRICE_MOVE]
        assert len(price_alerts) == 0


# ---------------------------------------------------------------------------
# AutonomousMonitor.check — breaking news
# ---------------------------------------------------------------------------


class TestBreakingNews:
    async def test_high_sentiment_triggers_alert(self, watchlist) -> None:
        articles = {"AAPL": [_make_article("AAPL", "FDA Approves New Product", sentiment=0.9)]}
        registry = _make_registry(news_provider=FakeNewsProvider(articles))
        monitor = AutonomousMonitor(watchlist, registry, check_interval=0, cooldown_minutes=0)

        alerts = await monitor.check()
        news_alerts = [a for a in alerts if a.trigger_type == TriggerType.BREAKING_NEWS]
        assert len(news_alerts) == 1
        assert news_alerts[0].ticker == "AAPL"
        assert "FDA" in news_alerts[0].summary

    async def test_low_sentiment_no_alert(self, watchlist) -> None:
        articles = {"AAPL": [_make_article("AAPL", "Routine update", sentiment=0.3)]}
        registry = _make_registry(news_provider=FakeNewsProvider(articles))
        monitor = AutonomousMonitor(watchlist, registry, check_interval=0, cooldown_minutes=0)

        alerts = await monitor.check()
        news_alerts = [a for a in alerts if a.trigger_type == TriggerType.BREAKING_NEWS]
        assert len(news_alerts) == 0

    async def test_negative_sentiment_triggers(self, watchlist) -> None:
        articles = {"TSLA": [_make_article("TSLA", "Major recall announced", sentiment=-0.8)]}
        registry = _make_registry(news_provider=FakeNewsProvider(articles))
        monitor = AutonomousMonitor(watchlist, registry, check_interval=0, cooldown_minutes=0)

        alerts = await monitor.check()
        news_alerts = [a for a in alerts if a.trigger_type == TriggerType.BREAKING_NEWS]
        assert len(news_alerts) == 1
        assert "negative" in news_alerts[0].summary

    async def test_no_news_provider_no_crash(self, watchlist) -> None:
        registry = _make_registry()
        monitor = AutonomousMonitor(watchlist, registry, check_interval=0)

        alerts = await monitor.check()
        assert len(alerts) == 0

    async def test_news_fetch_failure(self, watchlist) -> None:
        registry = _make_registry(news_provider=FailingNewsProvider())
        monitor = AutonomousMonitor(watchlist, registry, check_interval=0)

        alerts = await monitor.check()
        assert len(alerts) == 0

    async def test_none_sentiment_ignored(self, watchlist) -> None:
        articles = {"AAPL": [_make_article("AAPL", "Some news", sentiment=None)]}
        registry = _make_registry(news_provider=FakeNewsProvider(articles))
        monitor = AutonomousMonitor(watchlist, registry, check_interval=0, cooldown_minutes=0)

        alerts = await monitor.check()
        news_alerts = [a for a in alerts if a.trigger_type == TriggerType.BREAKING_NEWS]
        assert len(news_alerts) == 0


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------


class TestCooldown:
    async def test_cooldown_prevents_repeat_alert(self, watchlist) -> None:
        provider = FakePriceProvider({"AAPL": 200.0, "TSLA": 300.0})
        registry = _make_registry(price_provider=provider)
        monitor = AutonomousMonitor(watchlist, registry, check_interval=0, cooldown_minutes=60)

        await monitor.check()
        provider._prices["AAPL"] = 210.0
        alerts1 = await monitor.check()
        assert len([a for a in alerts1 if a.ticker == "AAPL"]) == 1

        # Another big move — still in cooldown
        provider._prices["AAPL"] = 230.0
        alerts2 = await monitor.check()
        assert len([a for a in alerts2 if a.ticker == "AAPL"]) == 0

    async def test_other_ticker_not_affected(self, watchlist) -> None:
        provider = FakePriceProvider({"AAPL": 200.0, "TSLA": 300.0})
        registry = _make_registry(price_provider=provider)
        monitor = AutonomousMonitor(watchlist, registry, check_interval=0, cooldown_minutes=60)

        await monitor.check()
        provider._prices["AAPL"] = 210.0
        await monitor.check()

        # TSLA should still be free to alert
        provider._prices["TSLA"] = 320.0  # ~6.7%
        alerts = await monitor.check()
        tsla_alerts = [a for a in alerts if a.ticker == "TSLA"]
        assert len(tsla_alerts) == 1


# ---------------------------------------------------------------------------
# Empty watchlist
# ---------------------------------------------------------------------------


class TestEmptyWatchlist:
    async def test_empty_watchlist_no_alerts(self, empty_watchlist) -> None:
        registry = _make_registry(FakePriceProvider({}))
        monitor = AutonomousMonitor(empty_watchlist, registry, check_interval=0)

        alerts = await monitor.check()
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# AutonomousAlert dataclass
# ---------------------------------------------------------------------------


class TestAutonomousAlert:
    def test_alert_fields(self) -> None:
        alert = AutonomousAlert(
            ticker="AAPL",
            trigger_type=TriggerType.PRICE_MOVE,
            summary="AAPL moved up 5%",
            severity=Severity.CRITICAL,
            data={"pct_change": 5.0},
        )
        assert alert.ticker == "AAPL"
        assert alert.trigger_type == TriggerType.PRICE_MOVE
        assert alert.severity == Severity.CRITICAL
        assert alert.data["pct_change"] == 5.0
        assert alert.created_at  # has a timestamp


# ---------------------------------------------------------------------------
# AutonomousAlertStore
# ---------------------------------------------------------------------------


def _alert(
    ticker: str = "AAPL",
    *,
    summary: str = "moved",
    severity: Severity = Severity.INFO,
    trigger_type: TriggerType = TriggerType.PRICE_MOVE,
    created_at: str | None = None,
    data: dict | None = None,
) -> AutonomousAlert:
    """Build an AutonomousAlert with optional overrides for timestamp/data."""
    kwargs: dict = {
        "ticker": ticker,
        "trigger_type": trigger_type,
        "summary": summary,
        "severity": severity,
        "data": data or {},
    }
    if created_at is not None:
        kwargs["created_at"] = created_at
    return AutonomousAlert(**kwargs)


class TestAutonomousAlertStore:
    def test_save_and_roundtrip(self, tmp_path) -> None:
        store = AutonomousAlertStore(tmp_path / "auto.json")
        alert = _alert("AAPL", summary="AAPL up 5%", severity=Severity.CRITICAL)
        store.save(alert)

        reloaded = AutonomousAlertStore(tmp_path / "auto.json")
        assert len(reloaded) == 1
        restored = reloaded.alerts[0]
        assert restored.ticker == "AAPL"
        assert restored.summary == "AAPL up 5%"
        assert restored.severity is Severity.CRITICAL
        assert restored.trigger_type is TriggerType.PRICE_MOVE

    def test_get_since_filters_by_timestamp(self, tmp_path) -> None:
        store = AutonomousAlertStore(tmp_path / "auto.json")
        now = datetime.now(timezone.utc)
        old = _alert(
            "OLD",
            summary="old alert",
            created_at=(now - timedelta(days=1)).isoformat(),
        )
        recent = _alert(
            "NEW",
            summary="recent alert",
            created_at=(now - timedelta(minutes=5)).isoformat(),
        )
        store.save(old)
        store.save(recent)

        since = now - timedelta(hours=1)
        results = store.get_since(since)
        assert len(results) == 1
        assert results[0].ticker == "NEW"

    def test_get_since_orders_newest_first(self, tmp_path) -> None:
        store = AutonomousAlertStore(tmp_path / "auto.json")
        now = datetime.now(timezone.utc)
        oldest = _alert(
            "A", summary="first", created_at=(now - timedelta(minutes=30)).isoformat()
        )
        middle = _alert(
            "B", summary="second", created_at=(now - timedelta(minutes=20)).isoformat()
        )
        newest = _alert(
            "C", summary="third", created_at=(now - timedelta(minutes=10)).isoformat()
        )
        # Save out of order to verify the store sorts on retrieval.
        store.save(middle)
        store.save(oldest)
        store.save(newest)

        results = store.get_since(now - timedelta(hours=1))
        assert [a.ticker for a in results] == ["C", "B", "A"]

    def test_get_since_skips_malformed_timestamp(self, tmp_path) -> None:
        store = AutonomousAlertStore(tmp_path / "auto.json")
        store.save(_alert("BAD", created_at="not-a-date"))
        store.save(_alert("GOOD"))

        results = store.get_since(datetime.now(timezone.utc) - timedelta(hours=1))
        tickers = {a.ticker for a in results}
        assert "GOOD" in tickers
        assert "BAD" not in tickers

    def test_max_alerts_cap(self, tmp_path) -> None:
        store = AutonomousAlertStore(tmp_path / "auto.json")
        store.MAX_ALERTS = 3  # type: ignore[misc]  # override for the test
        for i in range(5):
            store.save(_alert(f"T{i}"))

        tickers = [a.ticker for a in store.alerts]
        assert tickers == ["T2", "T3", "T4"]

    def test_clear(self, tmp_path) -> None:
        store = AutonomousAlertStore(tmp_path / "auto.json")
        store.save(_alert())
        store.save(_alert())
        store.clear()
        assert len(store) == 0
        # Clear is persisted.
        reloaded = AutonomousAlertStore(tmp_path / "auto.json")
        assert len(reloaded) == 0

    def test_load_handles_missing_file(self, tmp_path) -> None:
        store = AutonomousAlertStore(tmp_path / "missing.json")
        assert len(store) == 0
        assert store.get_since(datetime.now(timezone.utc)) == []

    def test_load_handles_malformed_file(self, tmp_path) -> None:
        path = tmp_path / "auto.json"
        path.write_text("not valid json", encoding="utf-8")
        store = AutonomousAlertStore(path)
        assert len(store) == 0

    def test_load_skips_malformed_entries(self, tmp_path) -> None:
        path = tmp_path / "auto.json"
        # One valid and one invalid entry — the valid one should still load.
        path.write_text(
            '[{"ticker": "OK", "trigger_type": "price_move", '
            '"summary": "ok", "severity": "info", "data": {}, '
            '"created_at": "2026-04-01T00:00:00+00:00"}, '
            '{"ticker": "BAD"}]',
            encoding="utf-8",
        )
        store = AutonomousAlertStore(path)
        assert len(store) == 1
        assert store.alerts[0].ticker == "OK"
