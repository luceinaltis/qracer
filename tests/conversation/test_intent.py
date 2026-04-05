"""Tests for IntentParser and Intent model."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from tracer.conversation.intent import (
    INTENT_TOOL_MAP,
    Intent,
    IntentParser,
    IntentType,
    _extract_tickers,
)
from tracer.llm.providers import CompletionResponse, Role
from tracer.llm.registry import LLMRegistry

# ---------------------------------------------------------------------------
# Intent dataclass
# ---------------------------------------------------------------------------


class TestIntent:
    def test_defaults_tools_from_map(self) -> None:
        intent = Intent(intent_type=IntentType.EVENT_ANALYSIS, raw_query="test")
        assert intent.tools == INTENT_TOOL_MAP[IntentType.EVENT_ANALYSIS]

    def test_explicit_tools_override(self) -> None:
        intent = Intent(
            intent_type=IntentType.EVENT_ANALYSIS,
            tools=["news"],
            raw_query="test",
        )
        assert intent.tools == ["news"]

    def test_frozen(self) -> None:
        intent = Intent(intent_type=IntentType.MACRO_QUERY)
        with pytest.raises(AttributeError):
            intent.intent_type = IntentType.DEEP_DIVE  # type: ignore[misc]

    def test_follow_up_has_empty_tools(self) -> None:
        intent = Intent(intent_type=IntentType.FOLLOW_UP)
        assert intent.tools == []


class TestExtractTickers:
    def test_extracts_uppercase(self) -> None:
        assert _extract_tickers("Why did AAPL spike?") == ["AAPL"]

    def test_multiple_tickers(self) -> None:
        assert _extract_tickers("Compare AAPL and TSLA") == ["AAPL", "TSLA"]

    def test_ignores_lowercase(self) -> None:
        assert _extract_tickers("the market is wild") == []

    def test_ignores_long_words(self) -> None:
        assert _extract_tickers("SOMETHING is not a ticker") == []

    def test_strips_punctuation(self) -> None:
        assert _extract_tickers("What about MSFT?") == ["MSFT"]


# ---------------------------------------------------------------------------
# IntentParser
# ---------------------------------------------------------------------------


def _make_registry(response_content: str) -> LLMRegistry:
    """Build an LLMRegistry with a mock researcher provider."""
    mock_provider = AsyncMock()
    mock_provider.complete.return_value = CompletionResponse(
        content=response_content,
        model="mock",
        input_tokens=10,
        output_tokens=5,
        cost=0.0,
    )
    registry = LLMRegistry()
    registry.register("mock", mock_provider, [Role.RESEARCHER])
    return registry


class TestIntentParserLLM:
    async def test_event_analysis(self) -> None:
        body = json.dumps({"intent": "event_analysis", "tickers": ["AAPL"]})
        parser = IntentParser(_make_registry(body))
        intent = await parser.parse("Why did AAPL spike 5% today?")
        assert intent.intent_type == IntentType.EVENT_ANALYSIS
        assert intent.tickers == ["AAPL"]
        assert intent.raw_query == "Why did AAPL spike 5% today?"

    async def test_deep_dive(self) -> None:
        body = json.dumps({"intent": "deep_dive", "tickers": ["TSMC"]})
        parser = IntentParser(_make_registry(body))
        intent = await parser.parse("Full analysis on TSMC")
        assert intent.intent_type == IntentType.DEEP_DIVE
        assert "fundamentals" in intent.tools

    async def test_macro_query(self) -> None:
        body = json.dumps({"intent": "macro_query", "tickers": []})
        parser = IntentParser(_make_registry(body))
        intent = await parser.parse("Where are we in the rate cycle?")
        assert intent.intent_type == IntentType.MACRO_QUERY
        assert intent.tools == ["macro"]

    async def test_alpha_hunt(self) -> None:
        body = json.dumps({"intent": "alpha_hunt", "tickers": []})
        parser = IntentParser(_make_registry(body))
        intent = await parser.parse("Where's the hidden alpha right now?")
        assert intent.intent_type == IntentType.ALPHA_HUNT

    async def test_cross_market(self) -> None:
        body = json.dumps({"intent": "cross_market", "tickers": []})
        parser = IntentParser(_make_registry(body))
        intent = await parser.parse("How does Korea semi data affect US AI stocks?")
        assert intent.intent_type == IntentType.CROSS_MARKET

    async def test_follow_up(self) -> None:
        body = json.dumps({"intent": "follow_up", "tickers": []})
        parser = IntentParser(_make_registry(body))
        intent = await parser.parse("What about insider trades?")
        assert intent.intent_type == IntentType.FOLLOW_UP


class TestIntentParserKeywordFallback:
    """When LLM fails, the parser falls back to keyword matching."""

    async def test_fallback_on_llm_error(self) -> None:
        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = RuntimeError("LLM down")
        registry = LLMRegistry()
        registry.register("mock", mock_provider, [Role.RESEARCHER])

        parser = IntentParser(registry)
        intent = await parser.parse("Why did AAPL spike?")
        assert intent.intent_type == IntentType.EVENT_ANALYSIS
        assert "AAPL" in intent.tickers

    async def test_keyword_deep_dive(self) -> None:
        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = RuntimeError("fail")
        registry = LLMRegistry()
        registry.register("mock", mock_provider, [Role.RESEARCHER])

        parser = IntentParser(registry)
        intent = await parser.parse("Full analysis on TSLA")
        assert intent.intent_type == IntentType.DEEP_DIVE

    async def test_keyword_alpha_hunt(self) -> None:
        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = RuntimeError("fail")
        registry = LLMRegistry()
        registry.register("mock", mock_provider, [Role.RESEARCHER])

        parser = IntentParser(registry)
        intent = await parser.parse("Where are the hidden opportunities?")
        assert intent.intent_type == IntentType.ALPHA_HUNT

    async def test_keyword_macro(self) -> None:
        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = RuntimeError("fail")
        registry = LLMRegistry()
        registry.register("mock", mock_provider, [Role.RESEARCHER])

        parser = IntentParser(registry)
        intent = await parser.parse("What is the current inflation rate?")
        assert intent.intent_type == IntentType.MACRO_QUERY

    async def test_keyword_cross_market(self) -> None:
        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = RuntimeError("fail")
        registry = LLMRegistry()
        registry.register("mock", mock_provider, [Role.RESEARCHER])

        parser = IntentParser(registry)
        intent = await parser.parse("How does oil affect airlines?")
        assert intent.intent_type == IntentType.CROSS_MARKET

    async def test_keyword_comparison_compare(self) -> None:
        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = RuntimeError("fail")
        registry = LLMRegistry()
        registry.register("mock", mock_provider, [Role.RESEARCHER])

        parser = IntentParser(registry)
        intent = await parser.parse("Compare AAPL and MSFT")
        assert intent.intent_type == IntentType.COMPARISON
        assert "AAPL" in intent.tickers
        assert "MSFT" in intent.tickers

    async def test_keyword_comparison_vs(self) -> None:
        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = RuntimeError("fail")
        registry = LLMRegistry()
        registry.register("mock", mock_provider, [Role.RESEARCHER])

        parser = IntentParser(registry)
        intent = await parser.parse("AAPL vs TSLA")
        assert intent.intent_type == IntentType.COMPARISON
        assert "AAPL" in intent.tickers
        assert "TSLA" in intent.tickers

    async def test_keyword_comparison_versus(self) -> None:
        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = RuntimeError("fail")
        registry = LLMRegistry()
        registry.register("mock", mock_provider, [Role.RESEARCHER])

        parser = IntentParser(registry)
        intent = await parser.parse("GOOG versus AMZN")
        assert intent.intent_type == IntentType.COMPARISON
        assert "GOOG" in intent.tickers
        assert "AMZN" in intent.tickers

    async def test_keyword_fallback_default(self) -> None:
        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = RuntimeError("fail")
        registry = LLMRegistry()
        registry.register("mock", mock_provider, [Role.RESEARCHER])

        parser = IntentParser(registry)
        intent = await parser.parse("ok")
        assert intent.intent_type == IntentType.FOLLOW_UP
