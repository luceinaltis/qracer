"""Price alerts — threshold-based notifications for tickers.

Persisted as ``alerts.json`` in the user's ``~/.qracer/`` directory.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AlertCondition(str, Enum):
    """Supported alert conditions."""

    ABOVE = "above"
    BELOW = "below"
    CHANGE_PCT = "change_pct"


@dataclass
class Alert:
    """A price alert for a ticker."""

    id: str
    ticker: str
    condition: AlertCondition
    threshold: float
    created_at: str
    active: bool = True
    triggered_at: str | None = None

    def evaluate(self, current_price: float, reference_price: float | None = None) -> bool:
        """Check whether this alert should trigger given the current price.

        Args:
            current_price: The current market price.
            reference_price: The price at alert creation time (needed for change_pct).

        Returns:
            True if the alert condition is met.
        """
        if self.condition == AlertCondition.ABOVE:
            return current_price > self.threshold
        if self.condition == AlertCondition.BELOW:
            return current_price < self.threshold
        if self.condition == AlertCondition.CHANGE_PCT:
            if reference_price is None or reference_price == 0:
                return False
            pct_change = ((current_price - reference_price) / reference_price) * 100
            return abs(pct_change) >= abs(self.threshold)
        return False

    def describe(self) -> str:
        """Return a human-readable description of this alert."""
        if self.condition == AlertCondition.ABOVE:
            return f"{self.ticker} goes above {self.threshold}"
        if self.condition == AlertCondition.BELOW:
            return f"{self.ticker} goes below {self.threshold}"
        return f"{self.ticker} changes by {self.threshold}%"


@dataclass
class AlertResult:
    """The outcome of a triggered alert."""

    alert: Alert
    triggered_price: float
    message: str


class AlertStore:
    """File-backed storage for price alerts.

    Usage::

        store = AlertStore(Path("~/.qracer/alerts.json"))
        alert = store.create("AAPL", AlertCondition.ABOVE, 200.0)
        active = store.get_active()
        store.mark_triggered(alert.id)
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._alerts: list[Alert] = self._load()

    @property
    def alerts(self) -> list[Alert]:
        """All alerts."""
        return list(self._alerts)

    def create(
        self,
        ticker: str,
        condition: AlertCondition,
        threshold: float,
        reference_price: float | None = None,
    ) -> Alert:
        """Create and persist a new alert."""
        alert = Alert(
            id=uuid.uuid4().hex[:8],
            ticker=ticker.upper(),
            condition=condition,
            threshold=threshold,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._alerts.append(alert)
        self._save()
        return alert

    def get_active(self) -> list[Alert]:
        """Return all active (not yet triggered) alerts."""
        return [a for a in self._alerts if a.active]

    def get_by_ticker(self, ticker: str) -> list[Alert]:
        """Return all alerts for a given ticker."""
        upper = ticker.upper()
        return [a for a in self._alerts if a.ticker == upper]

    def mark_triggered(self, alert_id: str, price: float | None = None) -> bool:
        """Mark an alert as triggered. Returns True if found and updated."""
        for alert in self._alerts:
            if alert.id == alert_id and alert.active:
                alert.active = False
                alert.triggered_at = datetime.now(timezone.utc).isoformat()
                self._save()
                return True
        return False

    def remove(self, alert_id: str) -> bool:
        """Remove an alert by ID. Returns True if found and removed."""
        for i, alert in enumerate(self._alerts):
            if alert.id == alert_id:
                self._alerts.pop(i)
                self._save()
                return True
        return False

    def clear(self) -> None:
        """Remove all alerts."""
        self._alerts.clear()
        self._save()

    def __len__(self) -> int:
        return len(self._alerts)

    def _load(self) -> list[Alert]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [self._deserialize(item) for item in data if isinstance(item, dict)]
        except (json.JSONDecodeError, OSError, KeyError, ValueError):
            logger.warning("Failed to load alerts from %s", self._path)
        return []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [self._serialize(a) for a in self._alerts]
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _serialize(alert: Alert) -> dict[str, Any]:
        d = asdict(alert)
        d["condition"] = alert.condition.value
        return d

    @staticmethod
    def _deserialize(data: dict[str, Any]) -> Alert:
        return Alert(
            id=data["id"],
            ticker=data["ticker"],
            condition=AlertCondition(data["condition"]),
            threshold=float(data["threshold"]),
            created_at=data["created_at"],
            active=data.get("active", True),
            triggered_at=data.get("triggered_at"),
        )
