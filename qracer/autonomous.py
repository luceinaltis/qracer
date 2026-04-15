"""Autonomous market monitoring during trading hours.

Watches watchlist tickers for significant price moves and breaking news,
sending notifications when thresholds are breached.  Designed to run
inside the ``Server`` tick loop alongside AlertMonitor and TaskExecutor.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from qracer.data.providers import NewsProvider, PriceProvider
from qracer.data.registry import DataRegistry
from qracer.watchlist import Watchlist

logger = logging.getLogger(__name__)

US_EASTERN = ZoneInfo("America/New_York")

# Defaults
DEFAULT_CHECK_INTERVAL = 60  # seconds between autonomous checks
DEFAULT_PRICE_THRESHOLD_PCT = 2.0
DEFAULT_COOLDOWN_MINUTES = 30


class TriggerType(str, Enum):
    PRICE_MOVE = "price_move"
    BREAKING_NEWS = "breaking_news"
    VOLUME_SPIKE = "volume_spike"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class AutonomousAlert:
    """A monitoring alert produced by the autonomous scanner."""

    ticker: str
    trigger_type: TriggerType
    summary: str
    severity: Severity
    data: dict[str, object] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def is_market_hours(now: datetime | None = None) -> bool:
    """Return True if *now* falls within US equity market hours.

    Market hours: 09:30–16:00 ET, Monday–Friday.
    """
    now = now or datetime.now(timezone.utc)
    et = now.astimezone(US_EASTERN)
    # Weekday: 0=Mon … 4=Fri
    if et.weekday() > 4:
        return False
    market_open = et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= et <= market_close


def _severity_for_pct(pct: float) -> Severity:
    """Map absolute percentage change to a severity level."""
    abs_pct = abs(pct)
    if abs_pct >= 5.0:
        return Severity.CRITICAL
    if abs_pct >= 3.0:
        return Severity.WARNING
    return Severity.INFO


class AutonomousMonitor:
    """Monitors watchlist tickers for price moves and breaking news.

    Follows the same ``should_check()`` / ``check()`` polling pattern
    used by :class:`AlertMonitor` and :class:`TaskExecutor`.

    Usage::

        monitor = AutonomousMonitor(watchlist, data_registry)
        if monitor.should_check():
            alerts = await monitor.check()
    """

    def __init__(
        self,
        watchlist: Watchlist,
        data_registry: DataRegistry,
        *,
        check_interval: float = DEFAULT_CHECK_INTERVAL,
        price_threshold_pct: float = DEFAULT_PRICE_THRESHOLD_PCT,
        cooldown_minutes: int = DEFAULT_COOLDOWN_MINUTES,
    ) -> None:
        self._watchlist = watchlist
        self._data = data_registry
        self._check_interval = check_interval
        self._price_threshold_pct = price_threshold_pct
        self._cooldown_seconds = cooldown_minutes * 60
        self._last_check: float | None = None
        # ticker → monotonic timestamp of last alert
        self._cooldowns: dict[str, float] = {}
        # ticker → last known price (for change detection)
        self._baseline_prices: dict[str, float] = {}

    def should_check(self) -> bool:
        """Return True when a check is due and the market is open."""
        if not is_market_hours():
            return False
        if self._last_check is None:
            return True
        return (time.monotonic() - self._last_check) >= self._check_interval

    async def check(self) -> list[AutonomousAlert]:
        """Scan watchlist tickers and return any triggered alerts."""
        self._last_check = time.monotonic()
        tickers = self._watchlist.tickers
        if not tickers:
            return []

        alerts: list[AutonomousAlert] = []
        price_alerts = await self._check_price_moves(tickers)
        alerts.extend(price_alerts)

        news_alerts = await self._check_breaking_news(tickers)
        alerts.extend(news_alerts)

        return alerts

    async def _check_price_moves(self, tickers: list[str]) -> list[AutonomousAlert]:
        """Detect significant price moves relative to the baseline."""
        try:
            provider: PriceProvider = self._data.get(PriceProvider)
        except KeyError:
            logger.debug("No PriceProvider — skipping price move check")
            return []

        alerts: list[AutonomousAlert] = []
        for ticker in tickers:
            if self._is_cooling_down(ticker):
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

            pct = ((price - baseline) / baseline) * 100
            if abs(pct) >= self._price_threshold_pct:
                direction = "up" if pct > 0 else "down"
                severity = _severity_for_pct(pct)
                alert = AutonomousAlert(
                    ticker=ticker,
                    trigger_type=TriggerType.PRICE_MOVE,
                    summary=(
                        f"{ticker} moved {direction} {abs(pct):.1f}%"
                        f" (${baseline:.2f} → ${price:.2f})"
                    ),
                    severity=severity,
                    data={"baseline": baseline, "current": price, "pct_change": round(pct, 2)},
                )
                alerts.append(alert)
                self._set_cooldown(ticker)
                # Reset baseline after alert
                self._baseline_prices[ticker] = price

        return alerts

    async def _check_breaking_news(self, tickers: list[str]) -> list[AutonomousAlert]:
        """Detect breaking news with strong sentiment."""
        try:
            provider: NewsProvider = self._data.get(NewsProvider)
        except KeyError:
            logger.debug("No NewsProvider — skipping news check")
            return []

        alerts: list[AutonomousAlert] = []
        for ticker in tickers:
            if self._is_cooling_down(ticker):
                continue
            try:
                articles = await provider.get_news(ticker, limit=3)
            except Exception:
                logger.debug("News fetch failed for %s", ticker)
                continue

            for article in articles:
                if article.sentiment is not None and abs(article.sentiment) >= 0.7:
                    tone = "positive" if article.sentiment > 0 else "negative"
                    severity = Severity.WARNING if abs(article.sentiment) >= 0.9 else Severity.INFO
                    alert = AutonomousAlert(
                        ticker=ticker,
                        trigger_type=TriggerType.BREAKING_NEWS,
                        summary=f"{ticker}: {article.title} (sentiment: {tone})",
                        severity=severity,
                        data={
                            "title": article.title,
                            "source": article.source,
                            "sentiment": article.sentiment,
                            "url": article.url,
                        },
                    )
                    alerts.append(alert)
                    self._set_cooldown(ticker)
                    break  # one news alert per ticker per cycle

        return alerts

    def _is_cooling_down(self, ticker: str) -> bool:
        """Return True if *ticker* was alerted recently."""
        last = self._cooldowns.get(ticker)
        if last is None:
            return False
        return (time.monotonic() - last) < self._cooldown_seconds

    def _set_cooldown(self, ticker: str) -> None:
        """Record the current time as the last alert for *ticker*."""
        self._cooldowns[ticker] = time.monotonic()


class AutonomousAlertStore:
    """File-backed storage for triggered autonomous alerts.

    Persists every :class:`AutonomousAlert` produced by
    :class:`AutonomousMonitor` so overnight findings can be surfaced on the
    next :command:`qracer repl` start via the session briefing.

    Usage::

        store = AutonomousAlertStore(Path("~/.qracer/autonomous_alerts.json"))
        store.save(alert)
        overnight = store.get_since(last_session)
    """

    # Keep the on-disk file bounded so a long-running ``qracer serve`` doesn't
    # grow the log indefinitely.  New alerts push the oldest ones out once the
    # cap is reached.
    MAX_ALERTS = 500

    def __init__(self, path: Path) -> None:
        self._path = path
        self._mtime: float = 0.0
        self._alerts: list[AutonomousAlert] = self._load()

    @property
    def alerts(self) -> list[AutonomousAlert]:
        """Return a copy of all persisted alerts."""
        self._maybe_reload()
        return list(self._alerts)

    def save(self, alert: AutonomousAlert) -> None:
        """Append *alert* to the store and flush to disk."""
        self._maybe_reload()
        self._alerts.append(alert)
        if len(self._alerts) > self.MAX_ALERTS:
            # Drop the oldest entries; ``created_at`` is monotonic per-process.
            self._alerts = self._alerts[-self.MAX_ALERTS :]
        self._save()

    def get_since(self, since: datetime) -> list[AutonomousAlert]:
        """Return alerts with ``created_at`` strictly after *since*.

        Alerts with a malformed or missing timestamp are skipped.  The
        returned list is ordered newest first, mirroring briefing output.
        """
        self._maybe_reload()
        out: list[tuple[datetime, AutonomousAlert]] = []
        for alert in self._alerts:
            dt = _parse_isoformat(alert.created_at)
            if dt is None:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt > since:
                out.append((dt, alert))
        out.sort(key=lambda item: item[0], reverse=True)
        return [alert for _, alert in out]

    def clear(self) -> None:
        """Remove all persisted alerts."""
        self._alerts.clear()
        self._save()

    def __len__(self) -> int:
        return len(self._alerts)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _maybe_reload(self) -> None:
        """Re-read from disk if another process modified the file."""
        if not self._path.exists():
            return
        try:
            current_mtime = self._path.stat().st_mtime
        except OSError:
            return
        if current_mtime != self._mtime:
            self._alerts = self._load()

    def _load(self) -> list[AutonomousAlert]:
        if not self._path.exists():
            return []
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._mtime = self._path.stat().st_mtime
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load autonomous alerts from %s", self._path)
            return []
        if not isinstance(raw, list):
            return []
        out: list[AutonomousAlert] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                out.append(self._deserialize(item))
            except (KeyError, ValueError):
                logger.debug("Skipping malformed autonomous alert record: %r", item)
        return out

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [self._serialize(a) for a in self._alerts]
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        try:
            self._mtime = self._path.stat().st_mtime
        except OSError:
            self._mtime = 0.0

    @staticmethod
    def _serialize(alert: AutonomousAlert) -> dict[str, Any]:
        d = asdict(alert)
        d["trigger_type"] = alert.trigger_type.value
        d["severity"] = alert.severity.value
        return d

    @staticmethod
    def _deserialize(data: dict[str, Any]) -> AutonomousAlert:
        return AutonomousAlert(
            ticker=str(data["ticker"]),
            trigger_type=TriggerType(data["trigger_type"]),
            summary=str(data["summary"]),
            severity=Severity(data["severity"]),
            data=dict(data.get("data", {}) or {}),
            created_at=str(data.get("created_at", "")),
        )


def _parse_isoformat(value: str) -> datetime | None:
    """Return ``datetime.fromisoformat(value)`` or ``None`` on failure."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
