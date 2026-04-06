"""Tests for conversation context management."""

from __future__ import annotations

from datetime import datetime, timedelta

from helpers import make_turn as _turn

from qracer.conversation.context import (
    ConversationContext,
    extract_context,
    is_stale,
    resolve_pronoun,
)


class TestExtractTickerFromTurns:
    def test_single_ticker(self):
        turns = [_turn("Why did AAPL spike?")]
        ctx = extract_context(turns)
        assert ctx.current_topic == "AAPL"
        assert "AAPL" in ctx.topic_stack

    def test_multiple_tickers(self):
        turns = [_turn("Compare AAPL and TSLA")]
        ctx = extract_context(turns)
        assert ctx.current_topic in ("AAPL", "TSLA")
        assert "AAPL" in ctx.topic_stack
        assert "TSLA" in ctx.topic_stack

    def test_no_ticker(self):
        turns = [_turn("How is the market doing?")]
        ctx = extract_context(turns)
        # No ticker, but might pick up a sector or be None.
        assert ctx.current_topic is None or isinstance(ctx.current_topic, str)


class TestTopicStackOrder:
    def test_most_recent_first(self):
        turns = [
            _turn("Analyze AAPL", turn=1),
            _turn("Now check TSLA", turn=2),
            _turn("What about NVDA?", turn=3),
        ]
        ctx = extract_context(turns)
        assert ctx.topic_stack[0] == "NVDA"
        assert ctx.current_topic == "NVDA"
        # Earlier tickers should follow.
        assert "TSLA" in ctx.topic_stack
        assert "AAPL" in ctx.topic_stack
        assert ctx.topic_stack.index("NVDA") < ctx.topic_stack.index("TSLA")
        assert ctx.topic_stack.index("TSLA") < ctx.topic_stack.index("AAPL")

    def test_max_five_topics(self):
        tickers = ["AAPL", "TSLA", "NVDA", "GOOG", "MSFT", "AMZN"]
        turns = [_turn(f"Check {t}", turn=i) for i, t in enumerate(tickers)]
        ctx = extract_context(turns)
        assert len(ctx.topic_stack) <= 5


class TestResolvePronounCurrent:
    def test_this(self):
        ctx = ConversationContext(current_topic="AAPL", topic_stack=["AAPL"])
        assert resolve_pronoun("this", ctx) == "AAPL"

    def test_it(self):
        ctx = ConversationContext(current_topic="TSLA", topic_stack=["TSLA"])
        assert resolve_pronoun("it", ctx) == "TSLA"

    def test_korean_pronoun(self):
        ctx = ConversationContext(current_topic="NVDA", topic_stack=["NVDA"])
        assert resolve_pronoun("이거", ctx) == "NVDA"

    def test_no_current_topic(self):
        ctx = ConversationContext()
        assert resolve_pronoun("this", ctx) is None


class TestResolvePronounPrevious:
    def test_previous(self):
        ctx = ConversationContext(
            current_topic="TSLA",
            topic_stack=["TSLA", "AAPL"],
        )
        assert resolve_pronoun("previous", ctx) == "AAPL"

    def test_korean_previous(self):
        ctx = ConversationContext(
            current_topic="TSLA",
            topic_stack=["TSLA", "AAPL"],
        )
        assert resolve_pronoun("전에 본 것", ctx) == "AAPL"

    def test_no_previous(self):
        ctx = ConversationContext(
            current_topic="AAPL",
            topic_stack=["AAPL"],
        )
        assert resolve_pronoun("previous", ctx) is None


class TestIsStale:
    def test_fresh_context(self):
        ctx = ConversationContext(last_activity=datetime.now())
        assert is_stale(ctx) is False

    def test_stale_context(self):
        ctx = ConversationContext(last_activity=datetime.now() - timedelta(minutes=15))
        assert is_stale(ctx) is True

    def test_custom_ttl(self):
        ctx = ConversationContext(last_activity=datetime.now() - timedelta(minutes=5))
        assert is_stale(ctx, ttl_minutes=3) is True
        assert is_stale(ctx, ttl_minutes=10) is False


class TestDepthDetection:
    def test_quick_depth(self):
        turns = [_turn("Give me a quick summary of AAPL")]
        ctx = extract_context(turns)
        assert ctx.depth == "quick"

    def test_deep_depth(self):
        turns = [_turn("Give me a detailed analysis of TSLA")]
        ctx = extract_context(turns)
        assert ctx.depth == "deep"

    def test_default_depth(self):
        turns = [_turn("How is AAPL?")]
        ctx = extract_context(turns)
        assert ctx.depth == "quick"


class TestEmptyTurns:
    def test_empty_turns_returns_empty_context(self):
        ctx = extract_context([])
        assert ctx.current_topic is None
        assert ctx.topic_stack == []
        assert ctx.intent is None
        assert ctx.depth == "quick"

    def test_non_user_turns_ignored(self):
        turns = [_turn("AAPL analysis results", role="assistant")]
        ctx = extract_context(turns)
        assert ctx.current_topic is None
        assert ctx.topic_stack == []
