"""Tests for the FinnhubWebSocketAdapter.

These exercise the streaming lifecycle (connect, subscribe, dispatch,
disconnect) with a fake WebSocket so the real network is never touched.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import patch

import pytest

from qracer.data.providers import NewsArticle


class FakeWebSocket:
    """A minimal async iterator mimicking websockets.WebSocketClientProtocol."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False
        self._incoming: asyncio.Queue[str | None] = asyncio.Queue()

    async def send(self, payload: str) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        self.closed = True
        await self._incoming.put(None)

    def __aiter__(self) -> "FakeWebSocket":
        return self

    async def __anext__(self) -> str:
        message = await self._incoming.get()
        if message is None:
            raise StopAsyncIteration
        return message

    # Test helpers ------------------------------------------------------

    async def push(self, message: str) -> None:
        await self._incoming.put(message)

    async def push_end(self) -> None:
        await self._incoming.put(None)


@pytest.fixture
def fake_ws() -> FakeWebSocket:
    return FakeWebSocket()


@pytest.fixture
def make_adapter(fake_ws: FakeWebSocket):
    """Return a factory that builds an adapter wired to ``fake_ws``."""

    def _factory():
        from qracer.data import finnhub_ws as mod

        # The fake mimics websockets.connect's awaitable contract.
        async def _fake_connect(url: str) -> FakeWebSocket:  # noqa: ARG001
            return fake_ws

        patches = [
            patch.object(mod, "_HAS_WEBSOCKETS", True),
            patch.object(
                mod,
                "websockets",
                type("M", (), {"connect": staticmethod(_fake_connect)}),
                create=True,
            ),
        ]
        for p in patches:
            p.start()

        adapter = mod.FinnhubWebSocketAdapter(api_key="test-key")
        return adapter, patches

    return _factory


class TestFinnhubWebSocketAdapterInit:
    def test_missing_websockets_raises(self) -> None:
        from qracer.data import finnhub_ws as mod

        with patch.object(mod, "_HAS_WEBSOCKETS", False):
            with pytest.raises(ImportError, match="websockets is not installed"):
                mod.FinnhubWebSocketAdapter(api_key="test")

    def test_missing_api_key_raises(self) -> None:
        from qracer.data import finnhub_ws as mod

        with patch.object(mod, "_HAS_WEBSOCKETS", True):
            with pytest.raises(ValueError, match="FINNHUB_API_KEY is required"):
                mod.FinnhubWebSocketAdapter(api_key=None)
            with pytest.raises(ValueError, match="FINNHUB_API_KEY is required"):
                mod.FinnhubWebSocketAdapter(api_key="")


class TestFinnhubWebSocketAdapter:
    async def test_connect_and_subscribe(self, make_adapter: Any, fake_ws: FakeWebSocket) -> None:
        adapter, patches = make_adapter()
        try:
            await adapter.connect()
            assert adapter.is_connected

            await adapter.subscribe(["AAPL", "msft"])

            sent = [json.loads(m) for m in fake_ws.sent]
            assert {"type": "subscribe", "symbol": "AAPL"} in sent
            assert {"type": "subscribe", "symbol": "MSFT"} in sent

            # Repeat subscribe is a no-op.
            fake_ws.sent.clear()
            await adapter.subscribe(["AAPL"])
            assert fake_ws.sent == []
        finally:
            await fake_ws.push_end()
            await adapter.disconnect()
            for p in patches:
                p.stop()

    async def test_unsubscribe_removes_symbol(
        self, make_adapter: Any, fake_ws: FakeWebSocket
    ) -> None:
        adapter, patches = make_adapter()
        try:
            await adapter.connect()
            await adapter.subscribe(["AAPL"])
            fake_ws.sent.clear()

            await adapter.unsubscribe(["aapl"])
            assert fake_ws.sent == [json.dumps({"type": "unsubscribe", "symbol": "AAPL"})]
            # After unsubscribe, a fresh subscribe goes through again.
            fake_ws.sent.clear()
            await adapter.subscribe(["AAPL"])
            assert any("subscribe" in m for m in fake_ws.sent)
        finally:
            await fake_ws.push_end()
            await adapter.disconnect()
            for p in patches:
                p.stop()

    async def test_price_callback_dispatch(self, make_adapter: Any, fake_ws: FakeWebSocket) -> None:
        adapter, patches = make_adapter()
        received: list[tuple[str, float]] = []

        async def handler(ticker: str, price: float) -> None:
            received.append((ticker, price))

        try:
            adapter.on_price(handler)
            await adapter.connect()

            await fake_ws.push(
                json.dumps(
                    {
                        "type": "trade",
                        "data": [
                            {"s": "AAPL", "p": 199.5},
                            {"s": "MSFT", "p": 410.25},
                        ],
                    }
                )
            )
            # Give the background task a chance to run.
            await asyncio.sleep(0.05)
            assert ("AAPL", 199.5) in received
            assert ("MSFT", 410.25) in received
        finally:
            await fake_ws.push_end()
            await adapter.disconnect()
            for p in patches:
                p.stop()

    async def test_news_callback_dispatch(self, make_adapter: Any, fake_ws: FakeWebSocket) -> None:
        adapter, patches = make_adapter()
        received: list[NewsArticle] = []

        async def handler(article: NewsArticle) -> None:
            received.append(article)

        try:
            adapter.on_news(handler)
            await adapter.connect()

            await fake_ws.push(
                json.dumps(
                    {
                        "type": "news",
                        "data": [
                            {
                                "headline": "Breaking",
                                "source": "wire",
                                "datetime": 1_700_000_000,
                                "url": "https://example.com/a",
                                "summary": "Sample",
                            }
                        ],
                    }
                )
            )
            await asyncio.sleep(0.05)
            assert len(received) == 1
            assert received[0].title == "Breaking"
            assert received[0].source == "wire"
            assert received[0].url == "https://example.com/a"
        finally:
            await fake_ws.push_end()
            await adapter.disconnect()
            for p in patches:
                p.stop()

    async def test_malformed_message_is_skipped(
        self, make_adapter: Any, fake_ws: FakeWebSocket
    ) -> None:
        adapter, patches = make_adapter()
        received: list[tuple[str, float]] = []

        async def handler(ticker: str, price: float) -> None:
            received.append((ticker, price))

        try:
            adapter.on_price(handler)
            await adapter.connect()

            await fake_ws.push("not-json")
            await fake_ws.push(json.dumps({"type": "unknown"}))
            await fake_ws.push(json.dumps({"type": "trade", "data": [{"s": "AAPL", "p": 201.0}]}))
            await asyncio.sleep(0.05)
            assert received == [("AAPL", 201.0)]
        finally:
            await fake_ws.push_end()
            await adapter.disconnect()
            for p in patches:
                p.stop()

    async def test_subscribe_before_connect_queued_until_connect(
        self, make_adapter: Any, fake_ws: FakeWebSocket
    ) -> None:
        adapter, patches = make_adapter()
        try:
            # Subscribe before the socket exists: the adapter remembers the
            # ticker and sends it once connect() succeeds.
            await adapter.subscribe(["AAPL"])
            assert fake_ws.sent == []

            await adapter.connect()
            await asyncio.sleep(0)
            sent = [json.loads(m) for m in fake_ws.sent]
            assert {"type": "subscribe", "symbol": "AAPL"} in sent
        finally:
            await fake_ws.push_end()
            await adapter.disconnect()
            for p in patches:
                p.stop()

    async def test_disconnect_marks_not_connected(
        self, make_adapter: Any, fake_ws: FakeWebSocket
    ) -> None:
        adapter, patches = make_adapter()
        try:
            await adapter.connect()
            assert adapter.is_connected
            await adapter.disconnect()
            assert not adapter.is_connected
            assert fake_ws.closed
        finally:
            for p in patches:
                p.stop()
