"""Tests for unknown topic handling (4-tier search)."""

from __future__ import annotations

import pytest

from qracer.conversation.context import ConversationContext
from qracer.conversation.topic_resolver import TopicSource, resolve_unknown_topic
from qracer.memory.memory_searcher import MemorySearcher


def _ctx(topics: list[str] | None = None) -> ConversationContext:
    return ConversationContext(
        current_topic=topics[0] if topics else None,
        topic_stack=topics or [],
    )


class TestLocalSearch:
    async def test_exact_match_current_topic(self):
        ctx = _ctx(["AAPL", "TSLA"])
        result = await resolve_unknown_topic("How is AAPL doing?", ctx)
        assert result.resolved is True
        assert result.source == TopicSource.LOCAL
        assert result.topic == "AAPL"

    async def test_exact_match_any_topic(self):
        ctx = _ctx(["AAPL", "TSLA", "NVDA"])
        result = await resolve_unknown_topic("What about TSLA?", ctx)
        assert result.resolved is True
        assert result.topic == "TSLA"

    async def test_no_match(self):
        ctx = _ctx(["AAPL"])
        result = await resolve_unknown_topic("Tell me about Bitcoin", ctx)
        assert result.resolved is False

    async def test_empty_context(self):
        ctx = _ctx()
        result = await resolve_unknown_topic("What is GOOG?", ctx)
        assert result.resolved is False


class TestEmbeddingSearch:
    async def test_found_in_memory(self):
        ctx = _ctx()
        searcher = MemorySearcher()
        try:
            searcher._ensure_fts_loaded()
        except Exception:
            searcher.close()
            pytest.skip("DuckDB FTS extension unavailable")
        searcher.index_summary("sess_aapl", "# AAPLA Analysis\nAAPL beat earnings...")
        result = await resolve_unknown_topic(
            "AAPL quarterly results", ctx, memory_searcher=searcher
        )
        assert result.resolved is True
        assert result.source == TopicSource.EMBEDDING
        searcher.close()

    async def test_not_found_in_memory(self):
        ctx = _ctx()
        searcher = MemorySearcher()
        try:
            searcher._ensure_fts_loaded()
        except Exception:
            searcher.close()
            pytest.skip("DuckDB FTS extension unavailable")
        result = await resolve_unknown_topic("Unknown stock", ctx, memory_searcher=searcher)
        assert result.resolved is False
        searcher.close()


class TestWebSearch:
    async def test_web_search_result(self):
        ctx = _ctx()

        def fake_search(query: str) -> str:
            return "Bitcoin is a cryptocurrency..."

        result = await resolve_unknown_topic("What is Bitcoin?", ctx, web_search_fn=fake_search)
        assert result.resolved is True
        assert result.source == TopicSource.WEB

    async def test_web_search_fallback_to_admit(self):
        ctx = _ctx()
        result = await resolve_unknown_topic("ObscureTopic123", ctx)
        assert result.resolved is False
        assert result.source == TopicSource.UNKNOWN
        assert "not sure" in (result.fallback_message or "").lower()


class TestTierPriority:
    async def test_local_takes_priority_over_web(self):
        """If topic is in local context, use it even if web is also available."""
        ctx = _ctx(["AAPL"])

        def fake_search(query: str) -> str:
            return "Apple Inc is a tech company..."

        result = await resolve_unknown_topic("How is AAPL?", ctx, web_search_fn=fake_search)
        assert result.source == TopicSource.LOCAL
