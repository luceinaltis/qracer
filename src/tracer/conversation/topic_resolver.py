"""Unknown topic handling — 4-tier search hierarchy.

Tier 1: Local context (<100ms) — current topic stack, session memory
Tier 2: Embedding search (<500ms) — past conversation summaries
Tier 3: Web search (1-3s) — ticker/sector definitions, latest news
Tier 4: Admit — 'I'm not sure. Can you tell me more about it?'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from tracer.conversation.context import ConversationContext
from tracer.memory.memory_searcher import MemorySearcher, SearchResult

logger = logging.getLogger(__name__)


class TopicSource(str, Enum):
    """Where the topic was resolved from."""
    LOCAL = "local"
    EMBEDDING = "embedding"
    WEB = "web"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TopicResolution:
    """Result of attempting to resolve an unknown topic."""

    resolved: bool
    topic: str | None
    source: TopicSource
    summary: str | None = None
    fallback_message: str | None = None


async def resolve_unknown_topic(
    query: str,
    context: ConversationContext,
    memory_searcher: MemorySearcher | None = None,
    web_search_fn=None,
) -> TopicResolution:
    """Attempt to resolve an unknown topic through 4-tier search.

    Args:
        query: The user's original query
        context: Current conversation context (topic stack, etc.)
        memory_searcher: Optional MemorySearcher for embedding search
        web_search_fn: Optional callable(query) -> str for web search
    """
    # Tier 1: Local context lookup
    result = _local_search(query, context)
    if result.resolved:
        return result

    # Tier 2: Embedding/memory search
    if memory_searcher:
        result = _embedding_search(query, memory_searcher)
        if result.resolved:
            return result

    # Tier 3: Web search
    if web_search_fn:
        result = _web_search(query, web_search_fn)
        if result.resolved:
            return result

    # Tier 4: Admit unknown
    return TopicResolution(
        resolved=False,
        topic=None,
        source=TopicSource.UNKNOWN,
        fallback_message=(
            f"I'm not sure about '{query}'. Can you tell me more about it?"
        ),
    )


def _local_search(query: str, context: ConversationContext) -> TopicResolution:
    """Check if the query matches the current topic stack."""
    query_lower = query.lower()
    for topic in context.topic_stack:
        if topic.lower() in query_lower or query_lower in topic.lower():
            return TopicResolution(
                resolved=True,
                topic=topic,
                source=TopicSource.LOCAL,
                summary=f"Found '{topic}' in current conversation context.",
            )
    return TopicResolution(resolved=False, topic=None, source=TopicSource.LOCAL)


def _embedding_search(
    query: str, searcher: MemorySearcher
) -> TopicResolution:
    """Search past session summaries for the topic."""
    results = searcher.search(query, limit=1)
    if results:
        hit: SearchResult = results[0]
        return TopicResolution(
            resolved=True,
            topic=hit.session_id,
            source=TopicSource.EMBEDDING,
            summary=hit.summary[:200] if hit.summary else None,
        )
    return TopicResolution(resolved=False, topic=None, source=TopicSource.EMBEDDING)


def _web_search(query: str, search_fn) -> TopicResolution:
    """Attempt web search for the topic."""
    try:
        result_text = search_fn(query)
        if result_text and len(result_text) > 10:
            return TopicResolution(
                resolved=True,
                topic=query,
                source=TopicSource.WEB,
                summary=result_text[:500],
            )
    except Exception as exc:
        logger.debug("Web search failed: %s", exc)
    return TopicResolution(resolved=False, topic=None, source=TopicSource.WEB)
