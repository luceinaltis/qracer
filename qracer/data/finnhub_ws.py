"""FinnhubWebSocketAdapter — real-time price and news streaming via Finnhub.

Implements the :class:`~qracer.data.providers.StreamingProvider` capability.
When connected, the adapter maintains a single WebSocket connection to
``wss://ws.finnhub.io`` and fans out incoming trade and news messages to
callbacks registered through :meth:`on_price` and :meth:`on_news`.

Callers typically:

1. Construct the adapter with a Finnhub API key.
2. Register callbacks via :meth:`on_price` / :meth:`on_news`.
3. Call :meth:`connect` on startup — on failure, fall back to REST polling.
4. :meth:`subscribe` to tickers as the watchlist changes.
5. :meth:`disconnect` during shutdown.

The adapter is deliberately tolerant of transient issues: connection failures
in :meth:`connect` are re-raised so the caller can activate a REST fallback,
but individual parse errors in the receive loop are logged and skipped.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from qracer.data.providers import NewsArticle, NewsCallback, PriceCallback

try:
    import websockets  # pyright: ignore[reportMissingImports]
    from websockets.exceptions import (  # pyright: ignore[reportMissingImports]
        ConnectionClosed as _WebSocketsConnectionClosed,
    )

    _HAS_WEBSOCKETS = True
    _ConnectionClosedTypes: tuple[type[BaseException], ...] = (_WebSocketsConnectionClosed,)
except ImportError:  # pragma: no cover - exercised only when extra missing
    _HAS_WEBSOCKETS = False
    _ConnectionClosedTypes = ()


logger = logging.getLogger(__name__)

_WS_URL_TEMPLATE = "wss://ws.finnhub.io?token={api_key}"


class FinnhubWebSocketAdapter:
    """Real-time price and news streaming via the Finnhub WebSocket API."""

    WS_URL = _WS_URL_TEMPLATE

    def __init__(self, api_key: str | None = None) -> None:
        if not _HAS_WEBSOCKETS:
            raise ImportError(
                "websockets is not installed. Install it with: uv add 'qracer[streaming]'"
            )
        if not api_key:
            raise ValueError("FINNHUB_API_KEY is required. Get one at https://finnhub.io/register")

        self._api_key = api_key
        self._ws: Any | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._price_callbacks: list[PriceCallback] = []
        self._news_callbacks: list[NewsCallback] = []
        self._subscribed: set[str] = set()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the WebSocket connection and start the receive loop.

        Raises the underlying exception if the connection cannot be
        established so callers can fall back to REST polling.
        """
        if self._ws is not None:
            return

        url = self.WS_URL.format(api_key=self._api_key)
        logger.info("FinnhubWebSocketAdapter connecting")
        self._ws = await websockets.connect(url)
        self._receive_task = asyncio.create_task(self._receive_loop())

        # Re-subscribe to any tickers added before the connection existed
        # (e.g. if the caller wired up callbacks before awaiting connect).
        pending = list(self._subscribed)
        self._subscribed.clear()
        if pending:
            await self.subscribe(pending)

    async def disconnect(self) -> None:
        """Cancel the receive loop and close the WebSocket."""
        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except (asyncio.CancelledError, Exception):
                pass
            self._receive_task = None

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                logger.debug("Error closing Finnhub WebSocket", exc_info=True)
            self._ws = None

        logger.info("FinnhubWebSocketAdapter disconnected")

    @property
    def is_connected(self) -> bool:
        """True if :meth:`connect` has been called and the socket is open."""
        return self._ws is not None

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    async def subscribe(self, tickers: list[str]) -> None:
        """Subscribe to real-time trade updates for *tickers*."""
        for ticker in tickers:
            symbol = ticker.upper()
            if symbol in self._subscribed:
                continue
            if self._ws is not None:
                await self._ws.send(json.dumps({"type": "subscribe", "symbol": symbol}))
            self._subscribed.add(symbol)

    async def unsubscribe(self, tickers: list[str]) -> None:
        """Unsubscribe from real-time trade updates for *tickers*."""
        for ticker in tickers:
            symbol = ticker.upper()
            if symbol not in self._subscribed:
                continue
            if self._ws is not None:
                await self._ws.send(json.dumps({"type": "unsubscribe", "symbol": symbol}))
            self._subscribed.discard(symbol)

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_price(self, callback: PriceCallback) -> None:
        """Register a coroutine to receive ``(ticker, price)`` updates."""
        self._price_callbacks.append(callback)

    def on_news(self, callback: NewsCallback) -> None:
        """Register a coroutine to receive streaming :class:`NewsArticle`."""
        self._news_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Receive loop
    # ------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        """Consume messages from the WebSocket until closed."""
        ws = self._ws
        if ws is None:
            return

        try:
            async for message in ws:
                try:
                    await self._dispatch(message)
                except Exception:
                    logger.debug("Failed to dispatch WS message", exc_info=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if _ConnectionClosedTypes and isinstance(exc, _ConnectionClosedTypes):
                logger.info("Finnhub WebSocket closed")
            else:
                logger.warning("Finnhub WebSocket receive loop error", exc_info=True)

    async def _dispatch(self, raw_message: str | bytes) -> None:
        """Parse a single incoming message and dispatch to callbacks."""
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8")

        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            logger.debug("Ignoring non-JSON WS payload: %r", raw_message[:200])
            return

        msg_type = payload.get("type")
        data = payload.get("data", [])

        if msg_type == "trade" and isinstance(data, list):
            for trade in data:
                ticker = trade.get("s")
                price = trade.get("p")
                if not ticker or price is None:
                    continue
                for cb in self._price_callbacks:
                    await cb(str(ticker), float(price))

        elif msg_type == "news" and isinstance(data, list):
            for item in data:
                article = _parse_news_item(item)
                if article is None:
                    continue
                for cb in self._news_callbacks:
                    await cb(article)


def _parse_news_item(item: dict[str, Any]) -> NewsArticle | None:
    """Convert a raw Finnhub news payload into a :class:`NewsArticle`."""
    if not isinstance(item, dict):
        return None

    headline = item.get("headline") or ""
    if not headline:
        return None

    ts = item.get("datetime", 0)
    try:
        published_at = datetime.fromtimestamp(float(ts))
    except (TypeError, ValueError, OSError):
        published_at = datetime.now()

    return NewsArticle(
        title=str(headline),
        source=str(item.get("source", "finnhub")),
        published_at=published_at,
        url=str(item.get("url", "")),
        summary=str(item.get("summary", "")),
        sentiment=None,
    )
