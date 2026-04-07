"""IntentParser — classifies natural-language queries into structured intents.

Maps each query to an IntentType that determines which pipeline tools to invoke.
Uses the researcher role (default: Claude Haiku) for fast classification.
Supports context-aware pronoun resolution and confidence-based ambiguity handling.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from qracer.llm.providers import CompletionRequest, CompletionResponse, Message, Role
from qracer.llm.registry import LLMRegistry

if TYPE_CHECKING:
    from qracer.conversation.context import ConversationContext

logger = logging.getLogger(__name__)

# Confidence threshold below which the parser signals ambiguity.
CONFIDENCE_HIGH = 0.7


class IntentType(str, Enum):
    """Supported intent types from the conversational-layer spec."""

    # QuickPath intents (< 5s, 0-1 LLM calls)
    PRICE_CHECK = "price_check"
    QUICK_NEWS = "quick_news"
    PORTFOLIO_CHECK = "portfolio_check"
    # DeepPath intents (30-120s, 3-7 LLM calls)
    EVENT_ANALYSIS = "event_analysis"
    DEEP_DIVE = "deep_dive"
    ALPHA_HUNT = "alpha_hunt"
    MACRO_QUERY = "macro_query"
    CROSS_MARKET = "cross_market"
    FOLLOW_UP = "follow_up"
    COMPARISON = "comparison"
    # Alert intents
    ALERT_SET = "alert_set"
    ALERT_LIST = "alert_list"


# Intents that use the QuickPath (no AnalysisLoop).
QUICKPATH_INTENTS = frozenset(
    {
        IntentType.PRICE_CHECK,
        IntentType.QUICK_NEWS,
        IntentType.PORTFOLIO_CHECK,
        IntentType.ALERT_SET,
        IntentType.ALERT_LIST,
    }
)

# Which pipeline tools each intent invokes by default.
INTENT_TOOL_MAP: dict[IntentType, list[str]] = {
    IntentType.PRICE_CHECK: ["price_event"],
    IntentType.QUICK_NEWS: ["news"],
    IntentType.PORTFOLIO_CHECK: [],  # handled directly by engine, no pipeline tools
    IntentType.EVENT_ANALYSIS: ["price_event", "news", "insider", "cross_market"],
    IntentType.DEEP_DIVE: [
        "price_event",
        "fundamentals",
        "news",
        "insider",
        "cross_market",
    ],
    IntentType.ALPHA_HUNT: ["macro", "cross_market", "news"],
    IntentType.MACRO_QUERY: ["macro"],
    IntentType.CROSS_MARKET: ["cross_market", "macro", "price_event"],
    IntentType.FOLLOW_UP: [],  # resolved from session context
    IntentType.COMPARISON: ["price_event", "fundamentals", "news"],
    IntentType.ALERT_SET: [],  # handled by alert subsystem
    IntentType.ALERT_LIST: [],  # handled by alert subsystem
}


@dataclass(frozen=True)
class Intent:
    """Structured intent parsed from a user query."""

    intent_type: IntentType
    tickers: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    raw_query: str = ""
    confidence: float = 1.0
    ambiguous: bool = False
    candidates: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # If no explicit tools, fill from the default map.
        if not self.tools:
            object.__setattr__(self, "tools", list(INTENT_TOOL_MAP.get(self.intent_type, [])))


_SYSTEM_PROMPT = """\
You are a query classifier for a financial analysis system.
Given a user query, return a JSON object with:
- "intent": one of price_check, quick_news, portfolio_check, event_analysis, \
deep_dive, alpha_hunt, macro_query, cross_market, follow_up, comparison, \
alert_set, alert_list
- "tickers": list of stock tickers mentioned (uppercase, empty list if none)
- "confidence": float 0.0-1.0 indicating how confident you are in the classification

Rules:
- "What's AAPL at?", "Price of X", "How's X doing?" → price_check
- "Any news on X?", "News for X" → quick_news
- "How's my portfolio?", "Check my holdings", "Portfolio P&L" → portfolio_check
- "Why did X move/spike/drop" → event_analysis
- "Full analysis on X" or "Tell me about X" → deep_dive
- "Where's alpha" or "hidden opportunities" → alpha_hunt
- Questions about rates, inflation, GDP, macro → macro_query
- Questions relating two markets/regions → cross_market
- Short follow-ups referencing prior context → follow_up
- "Compare X and Y", "X vs Y" (multiple tickers) → comparison
- "Alert me when X hits $Y", "Set alert for X" → alert_set
- "Show my alerts", "List alerts" → alert_list

Context pronouns:
- "this", "it", "this stock" → refers to the current topic in context
- "previous one" → refers to the previous topic
- "another one" → suggests a different topic from history

If the query is ambiguous or could map to multiple intents, set confidence low \
and include "candidates" — a list of the top 2-3 possible intent names.

Return ONLY valid JSON, no explanation."""


class IntentParser:
    """Classifies user queries into structured Intent objects.

    Uses the researcher LLM role for fast, low-cost classification.
    Falls back to keyword matching if the LLM call fails.
    Supports context-aware pronoun resolution when a ConversationContext is provided.
    """

    def __init__(self, llm_registry: LLMRegistry) -> None:
        self._llm_registry = llm_registry

    async def parse(
        self,
        query: str,
        context: "ConversationContext | None" = None,
    ) -> Intent:
        """Parse a natural-language query into an Intent.

        Args:
            query: Raw user input.
            context: Optional conversation context for pronoun resolution.

        Attempts LLM-based classification first, falls back to keyword matching.
        Resolves pronouns from context and flags ambiguous intents.
        """
        try:
            intent = await self._parse_with_llm(query)
        except Exception:
            logger.warning("LLM intent parsing failed, falling back to keywords", exc_info=True)
            intent = self._parse_with_keywords(query)

        # Resolve pronouns from context when no tickers were extracted.
        if context is not None and not intent.tickers and context.current_topic:
            resolved_tickers = _resolve_tickers_from_context(query, context)
            if resolved_tickers:
                intent = Intent(
                    intent_type=intent.intent_type,
                    tickers=resolved_tickers,
                    tools=intent.tools,
                    raw_query=intent.raw_query,
                    confidence=intent.confidence,
                    ambiguous=intent.ambiguous,
                    candidates=intent.candidates,
                )

        return intent

    async def _parse_with_llm(self, query: str) -> Intent:
        provider = self._llm_registry.get(Role.RESEARCHER)
        request = CompletionRequest(
            messages=[
                Message(role="system", content=_SYSTEM_PROMPT),
                Message(role="user", content=query),
            ],
            max_tokens=256,
            temperature=0.0,
        )
        response: CompletionResponse = await provider.complete(request)
        parsed = json.loads(response.content)

        intent_type = IntentType(parsed["intent"])
        tickers = [t.upper() for t in parsed.get("tickers", [])]
        confidence = float(parsed.get("confidence", 1.0))
        candidates = parsed.get("candidates", [])
        ambiguous = confidence < CONFIDENCE_HIGH
        return Intent(
            intent_type=intent_type,
            tickers=tickers,
            raw_query=query,
            confidence=confidence,
            ambiguous=ambiguous,
            candidates=candidates,
        )

    def _parse_with_keywords(self, query: str) -> Intent:
        """Keyword-based fallback when LLM is unavailable."""
        q = query.lower()

        # Extract uppercase tickers (simple heuristic: 1-5 letter uppercase words).
        tickers = _extract_tickers(query)

        # QuickPath: portfolio check (no ticker needed).
        if any(w in q for w in ("portfolio", "my holdings", "my stocks", "p&l", "pnl", "holdings")):
            return Intent(IntentType.PORTFOLIO_CHECK, tickers=tickers, raw_query=query)

        # Alert intents.
        if any(w in q for w in ("set alert", "alert me", "notify me when", "alert when")):
            return Intent(IntentType.ALERT_SET, tickers=tickers, raw_query=query)
        if any(w in q for w in ("my alerts", "list alerts", "show alerts")):
            return Intent(IntentType.ALERT_LIST, tickers=tickers, raw_query=query)

        # QuickPath: simple price check (single ticker, short query).
        if tickers and any(
            w in q
            for w in ("price", "what's", "whats", "how's", "hows", "how much", "quote", "at?")
        ):
            return Intent(IntentType.PRICE_CHECK, tickers=tickers, raw_query=query)

        # QuickPath: news lookup.
        if tickers and any(w in q for w in ("news", "headlines", "latest on")):
            return Intent(IntentType.QUICK_NEWS, tickers=tickers, raw_query=query)

        # Comparison must be checked first — "Compare AAPL spike" should be comparison.
        if len(tickers) >= 2 and any(w in q for w in ("compare", " vs ", "versus")):
            return Intent(IntentType.COMPARISON, tickers=tickers, raw_query=query)

        if any(w in q for w in ("spike", "drop", "crash", "move", "why did", "fell", "rose")):
            return Intent(IntentType.EVENT_ANALYSIS, tickers=tickers, raw_query=query)
        if any(w in q for w in ("full analysis", "deep dive", "tell me about", "analyze")):
            return Intent(IntentType.DEEP_DIVE, tickers=tickers, raw_query=query)
        if any(w in q for w in ("alpha", "hidden", "opportunity", "opportunities")):
            return Intent(IntentType.ALPHA_HUNT, tickers=tickers, raw_query=query)
        if any(w in q for w in ("rate", "inflation", "gdp", "fed", "macro", "cycle")):
            return Intent(IntentType.MACRO_QUERY, tickers=tickers, raw_query=query)
        if any(w in q for w in ("cross", "affect", "impact", "correlation", "relationship")):
            return Intent(IntentType.CROSS_MARKET, tickers=tickers, raw_query=query)
        if any(w in q for w in ("compare", " vs ", "versus")):
            return Intent(IntentType.COMPARISON, tickers=tickers, raw_query=query)

        # Default to follow_up for short/ambiguous queries.
        return Intent(IntentType.FOLLOW_UP, tickers=tickers, raw_query=query)


def _extract_tickers(query: str) -> list[str]:
    """Extract likely stock tickers from a query string."""
    tickers: list[str] = []
    for word in query.split():
        clean = word.strip("?,!.;:()")
        if clean.isupper() and 1 <= len(clean) <= 5 and clean.isalpha():
            tickers.append(clean)
    return tickers


def _resolve_tickers_from_context(
    query: str,
    context: "ConversationContext",
) -> list[str]:
    """Resolve tickers from conversation context using pronoun matching.

    Checks the query for pronoun patterns (this/it, previous, another one)
    and maps them to topics in the conversation context.
    """
    from qracer.conversation.context import resolve_pronoun

    q = query.strip().lower()

    resolved = resolve_pronoun(q, context)
    if resolved is not None:
        return [resolved]

    # Also check if any pronoun appears as a substring of the query.
    from qracer.conversation.constants import (
        ANOTHER_PRONOUNS,
        CURRENT_PRONOUNS,
        PREVIOUS_PRONOUNS,
    )

    for pronoun_set in (CURRENT_PRONOUNS, PREVIOUS_PRONOUNS, ANOTHER_PRONOUNS):
        for p in pronoun_set:
            if p in q:
                resolved = resolve_pronoun(p, context)
                if resolved is not None:
                    return [resolved]

    return []
