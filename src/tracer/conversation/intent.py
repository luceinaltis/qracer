"""IntentParser — classifies natural-language queries into structured intents.

Maps each query to an IntentType that determines which pipeline tools to invoke.
Uses the researcher role (default: Claude Haiku) for fast classification.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum

from tracer.llm.providers import CompletionRequest, CompletionResponse, Message, Role
from tracer.llm.registry import LLMRegistry

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """Supported intent types from the conversational-layer spec."""

    EVENT_ANALYSIS = "event_analysis"
    DEEP_DIVE = "deep_dive"
    ALPHA_HUNT = "alpha_hunt"
    MACRO_QUERY = "macro_query"
    CROSS_MARKET = "cross_market"
    FOLLOW_UP = "follow_up"
    COMPARISON = "comparison"


# Which pipeline tools each intent invokes by default.
INTENT_TOOL_MAP: dict[IntentType, list[str]] = {
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
}


@dataclass(frozen=True)
class Intent:
    """Structured intent parsed from a user query."""

    intent_type: IntentType
    tickers: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    raw_query: str = ""

    def __post_init__(self) -> None:
        # If no explicit tools, fill from the default map.
        if not self.tools:
            object.__setattr__(self, "tools", list(INTENT_TOOL_MAP.get(self.intent_type, [])))


_SYSTEM_PROMPT = """\
You are a query classifier for a financial analysis system.
Given a user query, return a JSON object with:
- "intent": one of event_analysis, deep_dive, alpha_hunt, macro_query, \
cross_market, follow_up, comparison
- "tickers": list of stock tickers mentioned (uppercase, empty list if none)

Rules:
- "Why did X move/spike/drop" → event_analysis
- "Full analysis on X" or "Tell me about X" → deep_dive
- "Where's alpha" or "hidden opportunities" → alpha_hunt
- Questions about rates, inflation, GDP, macro → macro_query
- Questions relating two markets/regions → cross_market
- Short follow-ups referencing prior context → follow_up
- "Compare X and Y", "X vs Y" (multiple tickers) → comparison

Return ONLY valid JSON, no explanation."""


class IntentParser:
    """Classifies user queries into structured Intent objects.

    Uses the researcher LLM role for fast, low-cost classification.
    Falls back to keyword matching if the LLM call fails.
    """

    def __init__(self, llm_registry: LLMRegistry) -> None:
        self._llm_registry = llm_registry

    async def parse(self, query: str) -> Intent:
        """Parse a natural-language query into an Intent.

        Attempts LLM-based classification first, falls back to keyword matching.
        """
        try:
            return await self._parse_with_llm(query)
        except Exception:
            logger.warning("LLM intent parsing failed, falling back to keywords", exc_info=True)
            return self._parse_with_keywords(query)

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
        return Intent(intent_type=intent_type, tickers=tickers, raw_query=query)

    def _parse_with_keywords(self, query: str) -> Intent:
        """Keyword-based fallback when LLM is unavailable."""
        q = query.lower()

        # Extract uppercase tickers (simple heuristic: 1-5 letter uppercase words).
        tickers = _extract_tickers(query)

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
