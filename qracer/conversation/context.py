"""ConversationContext — tracks topics, intent, and depth across turns.

Extracts structured context from session turn history so the engine can
resolve pronouns, maintain topic stacks, and detect stale sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from qracer.conversation.constants import (
    ANOTHER_PRONOUNS,
    CURRENT_PRONOUNS,
    DEFAULT_TURNS,
    DEPTH_DEEP,
    DEPTH_QUICK,
    INTENT_KEYWORDS,
    MAX_TOPIC_STACK,
    PREVIOUS_PRONOUNS,
    SECTORS,
    TICKER_RE,
    TICKER_STOPWORDS,
)
from qracer.memory.session_logger import TurnRecord


@dataclass
class ConversationContext:
    """Structured context extracted from recent conversation turns."""

    current_topic: str | None = None
    topic_stack: list[str] = field(default_factory=list)
    intent: str | None = None  # 'buy' | 'sell' | 'research' | 'monitor' | None
    depth: str = "quick"
    last_activity: datetime = field(default_factory=datetime.now)


def extract_context(turns: list[TurnRecord]) -> ConversationContext:
    """Parse recent user turns and build a ConversationContext.

    Examines the last 20 user turns (reverse chronological) to extract
    tickers, sectors, intent, and depth signals.
    """
    user_turns = [t for t in turns if t.role == "user"][-DEFAULT_TURNS:]

    if not user_turns:
        return ConversationContext()

    # Build topic stack in reverse chronological order (most recent first).
    topic_stack: list[str] = []
    for turn in reversed(user_turns):
        tickers = _extract_tickers(turn.content)
        sectors = _extract_sectors(turn.content)
        for topic in tickers + sectors:
            if topic not in topic_stack:
                topic_stack.append(topic)
            if len(topic_stack) >= MAX_TOPIC_STACK:
                break
        if len(topic_stack) >= MAX_TOPIC_STACK:
            break

    # Detect intent from most recent turns first.
    intent: str | None = None
    for turn in reversed(user_turns):
        detected = _detect_intent(turn.content)
        if detected:
            intent = detected
            break

    # Detect depth from most recent turns first.
    depth = "quick"
    for turn in reversed(user_turns):
        detected_depth = _detect_depth(turn.content)
        if detected_depth:
            depth = detected_depth
            break

    # Parse last_activity from the most recent turn's timestamp.
    last_ts = user_turns[-1].ts
    try:
        last_activity = datetime.fromisoformat(last_ts)
    except (ValueError, TypeError):
        last_activity = datetime.now()

    return ConversationContext(
        current_topic=topic_stack[0] if topic_stack else None,
        topic_stack=topic_stack,
        intent=intent,
        depth=depth,
        last_activity=last_activity,
    )


def resolve_pronoun(pronoun: str, context: ConversationContext) -> str | None:
    """Resolve a pronoun to a topic using the conversation context.

    Returns None if the pronoun cannot be resolved.
    """
    p = pronoun.strip().lower()

    if p in CURRENT_PRONOUNS:
        return context.current_topic

    if p in PREVIOUS_PRONOUNS:
        if len(context.topic_stack) > 1:
            return context.topic_stack[1]
        return None

    if p in ANOTHER_PRONOUNS:
        # Suggest from stack excluding the current topic.
        for topic in context.topic_stack:
            if topic != context.current_topic:
                return topic
        return None

    return None


def is_stale(context: ConversationContext, ttl_minutes: int = 10) -> bool:
    """Return True if the context's last_activity exceeds the TTL."""
    elapsed = datetime.now() - context.last_activity
    return elapsed.total_seconds() > ttl_minutes * 60


def _extract_tickers(text: str) -> list[str]:
    """Extract likely stock tickers from text."""
    matches = TICKER_RE.findall(text)
    return [m for m in matches if m not in TICKER_STOPWORDS]


def _extract_sectors(text: str) -> list[str]:
    """Extract sector/theme keywords from text."""
    lower = text.lower()
    return [s for s in SECTORS if s in lower]


def _detect_intent(text: str) -> str | None:
    """Detect trading intent from text."""
    lower = text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return intent
    return None


def _detect_depth(text: str) -> str | None:
    """Detect analysis depth from text."""
    lower = text.lower()
    words = set(lower.split())
    if words & DEPTH_DEEP:
        return "deep"
    if words & DEPTH_QUICK:
        return "quick"
    return None
