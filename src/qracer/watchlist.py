"""Watchlist — manage a list of tickers for quick access.

Persisted as a simple JSON file in ``~/.qracer/watchlist.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Watchlist:
    """File-backed watchlist of stock tickers.

    Usage::

        wl = Watchlist(Path("~/.qracer/watchlist.json"))
        wl.add("AAPL")
        wl.add("TSLA")
        wl.remove("TSLA")
        print(wl.tickers)   # ["AAPL"]
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._tickers: list[str] = self._load()

    @property
    def tickers(self) -> list[str]:
        """Current watchlist tickers in insertion order."""
        return list(self._tickers)

    def add(self, ticker: str) -> bool:
        """Add a ticker. Returns True if newly added, False if already present."""
        upper = ticker.upper()
        if upper in self._tickers:
            return False
        self._tickers.append(upper)
        self._save()
        return True

    def remove(self, ticker: str) -> bool:
        """Remove a ticker. Returns True if removed, False if not found."""
        upper = ticker.upper()
        if upper not in self._tickers:
            return False
        self._tickers.remove(upper)
        self._save()
        return True

    def contains(self, ticker: str) -> bool:
        return ticker.upper() in self._tickers

    def clear(self) -> None:
        self._tickers.clear()
        self._save()

    def __len__(self) -> int:
        return len(self._tickers)

    def _load(self) -> list[str]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [t for t in data if isinstance(t, str)]
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load watchlist from %s", self._path)
        return []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._tickers, indent=2), encoding="utf-8")
