"""Tests for IntentParser and Intent model."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from helpers import make_single_role_registry

from qracer.conversation.context import ConversationContext
from qracer.conversation.intent import (
    INTENT_TOOL_MAP,
    Intent,
    IntentParser,
    IntentType,
    _extract_tickers,
)
from qracer.llm.providers import Role
from qracer.llm.registry import LLMRegistry

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


class TestIntentParserLLM:
    async def test_event_analysis(self) -> None:
        body = json.dumps({"intent": "event_analysis", "tickers": ["AAPL"]})
        parser = IntentParser(make_single_role_registry(Role.RESEARCHER, body))
        intent = await parser.parse("Why did AAPL spike 5% today?")
        assert intent.intent_type == IntentType.EVENT_ANALYSIS
        assert intent.tickers == ["AAPL"]
        assert intent.raw_query == "Why did AAPL spike 5% today?"

    async def test_deep_dive(self) -> None:
        body = json.dumps({"intent": "deep_dive", "tickers": ["TSMC"]})
        parser = IntentParser(make_single_role_registry(Role.RESEARCHER, body))
        intent = await parser.parse("Full analysis on TSMC")
        assert intent.intent_type == IntentType.DEEP_DIVE
        assert "fundamentals" in intent.tools

    async def test_macro_query(self) -> None:
        body = json.dumps({"intent": "macro_query", "tickers": []})
        parser = IntentParser(make_single_role_registry(Role.RESEARCHER, body))
        intent = await parser.parse("Where are we in the rate cycle?")
        assert intent.intent_type == IntentType.MACRO_QUERY
        assert intent.tools == ["macro"]

    async def test_alpha_hunt(self) -> None:
        body = json.dumps({"intent": "alpha_hunt", "tickers": []})
        parser = IntentParser(make_single_role_registry(Role.RESEARCHER, body))
        intent = await parser.parse("Where's the hidden alpha right now?")
        assert intent.intent_type == IntentType.ALPHA_HUNT

    async def test_cross_market(self) -> None:
        body = json.dumps({"intent": "cross_market", "tickers": []})
        parser = IntentParser(make_single_role_registry(Role.RESEARCHER, body))
        intent = await parser.parse("How does Korea semi data affect US AI stocks?")
        assert intent.intent_type == IntentType.CROSS_MARKET

    async def test_follow_up(self) -> None:
        body = json.dumps({"intent": "follow_up", "tickers": []})
        parser = IntentParser(make_single_role_registry(Role.RESEARCHER, body))
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

    async def test_keyword_fallback_default(self) -> None:
        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = RuntimeError("fail")
        registry = LLMRegistry()
        registry.register("mock", mock_provider, [Role.RESEARCHER])

        parser = IntentParser(registry)
        intent = await parser.parse("ok")
        assert intent.intent_type == IntentType.FOLLOW_UP


# ---------------------------------------------------------------------------
# Alert intent types
# ---------------------------------------------------------------------------


class TestAlertIntentTypes:
    def test_alert_set_in_quickpath(self) -> None:
        from qracer.conversation.intent import QUICKPATH_INTENTS

        assert IntentType.ALERT_SET in QUICKPATH_INTENTS

    def test_alert_list_in_quickpath(self) -> None:
        from qracer.conversation.intent import QUICKPATH_INTENTS

        assert IntentType.ALERT_LIST in QUICKPATH_INTENTS

    def test_alert_set_tool_map(self) -> None:
        intent = Intent(intent_type=IntentType.ALERT_SET, raw_query="test")
        assert intent.tools == []

    def test_alert_list_tool_map(self) -> None:
        intent = Intent(intent_type=IntentType.ALERT_LIST, raw_query="test")
        assert intent.tools == []

    async def test_keyword_alert_set(self) -> None:
        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = RuntimeError("fail")
        registry = LLMRegistry()
        registry.register("mock", mock_provider, [Role.RESEARCHER])

        parser = IntentParser(registry)
        intent = await parser.parse("Set alert for AAPL at $200")
        assert intent.intent_type == IntentType.ALERT_SET
        assert "AAPL" in intent.tickers

    async def test_keyword_alert_list(self) -> None:
        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = RuntimeError("fail")
        registry = LLMRegistry()
        registry.register("mock", mock_provider, [Role.RESEARCHER])

        parser = IntentParser(registry)
        intent = await parser.parse("Show my alerts")
        assert intent.intent_type == IntentType.ALERT_LIST

    async def test_llm_alert_set(self) -> None:
        body = json.dumps({"intent": "alert_set", "tickers": ["TSLA"], "confidence": 0.95})
        parser = IntentParser(make_single_role_registry(Role.RESEARCHER, body))
        intent = await parser.parse("Alert me when TSLA hits $300")
        assert intent.intent_type == IntentType.ALERT_SET
        assert intent.tickers == ["TSLA"]

    async def test_llm_alert_list(self) -> None:
        body = json.dumps({"intent": "alert_list", "tickers": [], "confidence": 0.9})
        parser = IntentParser(make_single_role_registry(Role.RESEARCHER, body))
        intent = await parser.parse("List my alerts")
        assert intent.intent_type == IntentType.ALERT_LIST


# ---------------------------------------------------------------------------
# Confidence and ambiguity
# ---------------------------------------------------------------------------


class TestConfidenceAndAmbiguity:
    def test_intent_default_confidence(self) -> None:
        intent = Intent(intent_type=IntentType.PRICE_CHECK, raw_query="test")
        assert intent.confidence == 1.0
        assert intent.ambiguous is False
        assert intent.candidates == []

    async def test_llm_high_confidence_not_ambiguous(self) -> None:
        body = json.dumps({"intent": "deep_dive", "tickers": ["AAPL"], "confidence": 0.9})
        parser = IntentParser(make_single_role_registry(Role.RESEARCHER, body))
        intent = await parser.parse("Full analysis on AAPL")
        assert intent.confidence == 0.9
        assert intent.ambiguous is False

    async def test_llm_low_confidence_is_ambiguous(self) -> None:
        body = json.dumps({
            "intent": "deep_dive",
            "tickers": ["AAPL"],
            "confidence": 0.4,
            "candidates": ["deep_dive", "event_analysis"],
        })
        parser = IntentParser(make_single_role_registry(Role.RESEARCHER, body))
        intent = await parser.parse("What about AAPL")
        assert intent.confidence == 0.4
        assert intent.ambiguous is True
        assert intent.candidates == ["deep_dive", "event_analysis"]

    async def test_confidence_threshold_boundary(self) -> None:
        body = json.dumps({"intent": "price_check", "tickers": ["MSFT"], "confidence": 0.7})
        parser = IntentParser(make_single_role_registry(Role.RESEARCHER, body))
        intent = await parser.parse("MSFT?")
        # Exactly at threshold should NOT be ambiguous
        assert intent.ambiguous is False

    async def test_just_below_threshold_is_ambiguous(self) -> None:
        body = json.dumps({
            "intent": "follow_up",
            "tickers": [],
            "confidence": 0.69,
            "candidates": ["follow_up", "deep_dive"],
        })
        parser = IntentParser(make_single_role_registry(Role.RESEARCHER, body))
        intent = await parser.parse("looks good")
        assert intent.ambiguous is True


# ---------------------------------------------------------------------------
# Context-aware pronoun resolution in IntentParser
# ---------------------------------------------------------------------------


class TestIntentParserPronounResolution:
    """IntentParser resolves pronouns when context is provided."""

    async def test_this_stock_resolves_to_current_topic(self) -> None:
        body = json.dumps({"intent": "deep_dive", "tickers": [], "confidence": 0.85})
        parser = IntentParser(make_single_role_registry(Role.RESEARCHER, body))
        ctx = ConversationContext(
            current_topic="AAPL",
            topic_stack=["AAPL", "TSLA"],
        )
        intent = await parser.parse("Tell me about this stock", context=ctx)
        assert intent.tickers == ["AAPL"]

    async def test_previous_one_resolves_to_stack(self) -> None:
        body = json.dumps({"intent": "deep_dive", "tickers": [], "confidence": 0.85})
        parser = IntentParser(make_single_role_registry(Role.RESEARCHER, body))
        ctx = ConversationContext(
            current_topic="AAPL",
            topic_stack=["AAPL", "TSLA", "NVDA"],
        )
        intent = await parser.parse("What about the previous one", context=ctx)
        assert intent.tickers == ["TSLA"]

    async def test_another_one_resolves_excluding_current(self) -> None:
        body = json.dumps({"intent": "deep_dive", "tickers": [], "confidence": 0.85})
        parser = IntentParser(make_single_role_registry(Role.RESEARCHER, body))
        ctx = ConversationContext(
            current_topic="AAPL",
            topic_stack=["AAPL", "TSLA", "NVDA"],
        )
        intent = await parser.parse("Show me another one", context=ctx)
        assert intent.tickers == ["TSLA"]

    async def test_it_resolves_current(self) -> None:
        body = json.dumps({"intent": "price_check", "tickers": [], "confidence": 0.8})
        parser = IntentParser(make_single_role_registry(Role.RESEARCHER, body))
        ctx = ConversationContext(
            current_topic="MSFT",
            topic_stack=["MSFT"],
        )
        intent = await parser.parse("How much is it?", context=ctx)
        assert intent.tickers == ["MSFT"]

    async def test_no_context_no_resolution(self) -> None:
        body = json.dumps({"intent": "follow_up", "tickers": [], "confidence": 0.5})
        parser = IntentParser(make_single_role_registry(Role.RESEARCHER, body))
        intent = await parser.parse("Tell me about this", context=None)
        assert intent.tickers == []

    async def test_explicit_ticker_not_overridden(self) -> None:
        body = json.dumps({"intent": "deep_dive", "tickers": ["NVDA"], "confidence": 0.9})
        parser = IntentParser(make_single_role_registry(Role.RESEARCHER, body))
        ctx = ConversationContext(
            current_topic="AAPL",
            topic_stack=["AAPL", "TSLA"],
        )
        intent = await parser.parse("Full analysis on NVDA", context=ctx)
        assert intent.tickers == ["NVDA"]

    async def test_korean_pronoun_resolves(self) -> None:
        body = json.dumps({"intent": "price_check", "tickers": [], "confidence": 0.85})
        parser = IntentParser(make_single_role_registry(Role.RESEARCHER, body))
        ctx = ConversationContext(
            current_topic="AAPL",
            topic_stack=["AAPL"],
        )
        intent = await parser.parse("이거 가격 얼마야?", context=ctx)
        assert intent.tickers == ["AAPL"]
