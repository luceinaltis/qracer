"""Autonomous market monitoring during trading hours.

Watches user watchlist tickers for significant price moves and breaking
news, emitting ``AutonomousAlert`` instances when thresholds are exceeded.
Designed to plug into the Server tick loop alongside AlertMonitor and
TaskExecutor.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from zoneinfo import ZoneInfo

from qracer.data.providers import NewsProvider, PriceProvider
from qracer.data.registry import DataRegistry
from qracer.watchlist import Watchlist

logger = logging.getLogger(__name__)

US_EASTERN = ZoneInfo("America/New_York")


class TriggerType(Enum):
    """What caused an autonomous alert."""

    PRICE_MOVE = "price_move"
    BREAKING_NEWS = "breaking_news"


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class AutonomousAlert:
    """A single autonomous-monitoring alert."""

    ticker: str
    trigger: TriggerType
    summary: str
    severity: Severity
    data: dict[str, object] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


def is_market_open(now: datetime | None = None) -> bool:
    """Return True if US equity markets are currently open.

    Market hours: 9:30 AM – 4:00 PM ET, Monday–Friday.
    """
    now = now or datetime.now(tz=timezone.utc)
    et = now.astimezone(US_EASTERN)
    # Weekends (Saturday=5, Sunday=6)
    if et.weekday() >= 5:
        return False
    market_open = et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= et < market_close


class AutonomousMonitor:
    """Watches watchlist tickers for significant events during market hours.

    Parameters
    ----------
    watchlist:
        Watchlist instance providing tickers to monitor.
    data_registry:
        Registry supplying PriceProvider and (optionally) NewsProvider.
    price_move_threshold_pct:
        Minimum percentage move to trigger a price alert (default 2%).
    alert_cooldown_minutes:
        Minimum minutes between alerts for the same ticker (default 30).
    check_interval:
        Seconds between successive scans (default 60).
    """

    def __init__(
        self,
        watchlist: Watchlist,
        data_registry: DataRegistry,
        *,
        price_move_threshold_pct: float = 2.0,
        alert_cooldown_minutes: int = 30,
        check_interval: float = 60.0,
    ) -> None:
        self._watchlist = watchlist
        self._data_registry = data_registry
        self._threshold_pct = price_move_threshold_pct
        self._cooldown_seconds = alert_cooldown_minutes * 60
        self._check_interval = check_interval

        self._last_check: float | None = None
        # ticker → last price used as baseline
        self._baseline_prices: dict[str, float] = {}
        # ticker → monotonic timestamp of last alert
        self._last_alert_time: dict[str, float] = {}

    def should_check(self) -> bool:
        """Return True if enough time has elapsed since the last check."""
        if self._last_check is None:
            return True
        return (time.monotonic() - self._last_check) >= self._check_interval

    def _is_on_cooldown(self, ticker: str) -> bool:
        last = self._last_alert_time.get(ticker)
        if last is None:
            return False
        return (time.monotonic() - last) < self._cooldown_seconds

    def _record_alert(self, ticker: str) -> None:
        self._last_alert_time[ticker] = time.monotonic()

    async def check(self) -> list[AutonomousAlert]:
        """Run one monitoring sweep.

        Returns a (possibly empty) list of alerts.  Only runs during US
        market hours; returns ``[]`` outside of trading sessions.
        """
        self._last_check = time.monotonic()

        if not is_market_open():
            logger.debug("Market closed — skipping autonomous check")
            return []

        tickers = self._watchlist.tickers
        if not tickers:
            return []

        alerts: list[AutonomousAlert] = []

        # Price-move detection
        alerts.extend(await self._check_price_moves(tickers))

        # Breaking-news detection
        alerts.extend(await self._check_breaking_news(tickers))

        return alerts

    async def _check_price_moves(self, tickers: list[str]) -> list[AutonomousAlert]:
        try:
            provider: PriceProvider = self._data_registry.get(PriceProvider)
        except KeyError:
            logger.debug("No PriceProvider — skipping price-move check")
            return []

        alerts: list[AutonomousAlert] = []
        for ticker in tickers:
            if self._is_on_cooldown(ticker):
                continue
            try:
                price = await provider.get_price(ticker)
            except Exception:
                logger.debug("Price fetch failed for %s", ticker)
                continue

            baseline = self._baseline_prices.get(ticker)
            self._baseline_prices[ticker] = price

            if baseline is None or baseline == 0:
                continue

            pct_change = ((price - baseline) / baseline) * 100

            if abs(pct_change) >= self._threshold_pct:
                direction = "up" if pct_change > 0 else "down"
                severity = Severity.HIGH if abs(pct_change) >= self._threshold_pct * 2 else Severity.MEDIUM
                alert = AutonomousAlert(
                    ticker=ticker,
                    trigger=TriggerType.PRICE_MOVE,
                    summary=f"{ticker} moved {direction} {abs(pct_change):.1f}% (${baseline:.2f} → ${price:.2f})",
                    severity=severity,
                    data={"baseline": baseline, "current": price, "pct_change": pct_change},
                )
                alerts.append(alert)
                self._record_alert(ticker)
                # Reset baseline so next move is measured from new price
                self._baseline_prices[ticker] = price

        return alerts

    async def _check_breaking_news(self, tickers: list[str]) -> list[AutonomousAlert]:
        try:
            provider: NewsProvider = self._data_registry.get(NewsProvider)
        except KeyError:
            logger.debug("No NewsProvider — skipping news check")
            return []

        alerts: list[AutonomousAlert] = []
        for ticker in tickers:
            if self._is_on_cooldown(ticker):
                continue
            try:
                articles = await provider.get_news(ticker, limit=5)
            except Exception:
                logger.debug("News fetch failed for %s", ticker)
                continue

            for article in articles:
                if article.sentiment is not None and abs(article.sentiment) >= 0.8:
                    severity = Severity.HIGH if abs(article.sentiment) >= 0.9 else Severity.MEDIUM
                    alert = AutonomousAlert(
                        ticker=ticker,
                        trigger=TriggerType.BREAKING_NEWS,
                        summary=f"Breaking: {article.title}",
                        severity=severity,
                        data={
                            "title": article.title,
                            "source": article.source,
                            "sentiment": article.sentiment,
                            "url": article.url,
                        },
                    )
                    alerts.append(alert)
                    self._record_alert(ticker)
                    break  # One news alert per ticker per sweep

        return alerts
