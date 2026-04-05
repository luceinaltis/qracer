"""ConversationContext — tracks topics, intent, and depth across turns.

Extracts structured context from session turn history so the engine can
resolve pronouns, maintain topic stacks, and detect stale sessions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from qracer.memory.session_logger import TurnRecord

# Uppercase 1-5 letter words that look like tickers.
_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")

# Common English words that match the ticker pattern but aren't tickers.
_TICKER_STOPWORDS = frozenset(
    {
        "I",
        "A",
        "AM",
        "AN",
        "AS",
        "AT",
        "BE",
        "BY",
        "DO",
        "GO",
        "HE",
        "IF",
        "IN",
        "IS",
        "IT",
        "ME",
        "MY",
        "NO",
        "OF",
        "OK",
        "ON",
        "OR",
        "SO",
        "TO",
        "UP",
        "US",
        "WE",
        "AND",
        "ARE",
        "BUT",
        "CAN",
        "DID",
        "FOR",
        "GET",
        "GOT",
        "HAS",
        "HAD",
        "HER",
        "HIM",
        "HIS",
        "HOW",
        "ITS",
        "LET",
        "MAY",
        "NOT",
        "NOW",
        "OLD",
        "OUR",
        "OWN",
        "SAY",
        "SHE",
        "THE",
        "TOO",
        "TRY",
        "USE",
        "WAY",
        "WHO",
        "WHY",
        "YOU",
        "ALL",
        "ANY",
        "DAY",
        "FEW",
        "NEW",
        "ALSO",
        "BACK",
        "BEEN",
        "CALL",
        "COME",
        "EACH",
        "FIND",
        "FROM",
        "GIVE",
        "GOOD",
        "HAVE",
        "HERE",
        "HIGH",
        "JUST",
        "KNOW",
        "LAST",
        "LIKE",
        "LONG",
        "LOOK",
        "MADE",
        "MAKE",
        "MORE",
        "MOST",
        "MUCH",
        "MUST",
        "NAME",
        "ONLY",
        "OVER",
        "PART",
        "SAME",
        "SHOW",
        "SOME",
        "SUCH",
        "TAKE",
        "TELL",
        "THAN",
        "THAT",
        "THEM",
        "THEN",
        "THEY",
        "THIS",
        "TIME",
        "VERY",
        "WANT",
        "WELL",
        "WERE",
        "WHAT",
        "WHEN",
        "WILL",
        "WITH",
        "WORK",
        "YEAR",
        "YOUR",
        "ABOUT",
        "AFTER",
        "BEING",
        "COULD",
        "EVERY",
        "FIRST",
        "GREAT",
        "NEVER",
        "OTHER",
        "PLACE",
        "POINT",
        "RIGHT",
        "SHALL",
        "SINCE",
        "SMALL",
        "STILL",
        "THEIR",
        "THERE",
        "THESE",
        "THING",
        "THINK",
        "THOSE",
        "THREE",
        "UNDER",
        "WHERE",
        "WHICH",
        "WHILE",
        "WORLD",
        "WOULD",
        "QUICK",
        "BRIEF",
        "DEEP",
        "BUY",
        "SELL",
    }
)

_SECTORS = frozenset(
    {
        "semiconductor",
        "tech",
        "technology",
        "energy",
        "finance",
        "financial",
        "healthcare",
        "biotech",
        "retail",
        "automotive",
        "ev",
        "battery",
        "ai",
        "cloud",
        "crypto",
        "real estate",
        "defense",
        "telecom",
        "media",
        "consumer",
        "industrial",
        "materials",
        "utilities",
    }
)

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "buy": ["buy", "long", "entry", "accumulate", "purchase"],
    "sell": ["sell", "short", "exit", "dump", "liquidate"],
    "research": ["analyze", "analysis", "research", "investigate", "look into", "tell me about"],
    "monitor": ["monitor", "watch", "track", "alert", "notify"],
}

_DEPTH_QUICK = {"quick", "brief", "summary", "briefly", "short", "overview", "glance"}
_DEPTH_DEEP = {"analyze", "detail", "detailed", "deep", "thorough", "comprehensive", "full"}

# Pronouns that resolve to current or previous topic.
_CURRENT_PRONOUNS = {"this", "it", "this stock", "이거", "이것"}
_PREVIOUS_PRONOUNS = {"previous", "previous one", "전에 본 것"}

_MAX_TOPIC_STACK = 5
_DEFAULT_TURNS = 20


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
    user_turns = [t for t in turns if t.role == "user"][-_DEFAULT_TURNS:]

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
            if len(topic_stack) >= _MAX_TOPIC_STACK:
                break
        if len(topic_stack) >= _MAX_TOPIC_STACK:
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

    if p in _CURRENT_PRONOUNS:
        return context.current_topic

    if p in _PREVIOUS_PRONOUNS:
        if len(context.topic_stack) > 1:
            return context.topic_stack[1]
        return None

    return None


def is_stale(context: ConversationContext, ttl_minutes: int = 10) -> bool:
    """Return True if the context's last_activity exceeds the TTL."""
    elapsed = datetime.now() - context.last_activity
    return elapsed.total_seconds() > ttl_minutes * 60


def _extract_tickers(text: str) -> list[str]:
    """Extract likely stock tickers from text."""
    matches = _TICKER_RE.findall(text)
    return [m for m in matches if m not in _TICKER_STOPWORDS]


def _extract_sectors(text: str) -> list[str]:
    """Extract sector/theme keywords from text."""
    lower = text.lower()
    return [s for s in _SECTORS if s in lower]


def _detect_intent(text: str) -> str | None:
    """Detect trading intent from text."""
    lower = text.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return intent
    return None


def _detect_depth(text: str) -> str | None:
    """Detect analysis depth from text."""
    lower = text.lower()
    words = set(lower.split())
    if words & _DEPTH_DEEP:
        return "deep"
    if words & _DEPTH_QUICK:
        return "quick"
    return None
