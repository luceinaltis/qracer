"""SessionCompactor — compresses conversation turns into a Markdown summary (Tier 2).

When a session exceeds a token threshold, the compactor sends the raw turns to
the reporter LLM role (Haiku) and receives a concise Markdown summary.  The
summary replaces raw turns in the active context window while the JSONL audit
log is preserved untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from tracer.llm.providers import CompletionRequest, Message, Role
from tracer.llm.registry import LLMRegistry
from tracer.memory.session_logger import SessionLogger, TurnRecord

logger = logging.getLogger(__name__)

# Default token threshold for triggering compaction (per spec: 8 000 tokens).
DEFAULT_TOKEN_THRESHOLD = 8_000

_SYSTEM_PROMPT = """\
You are a financial analysis session summariser.
Given a conversation log, produce a concise Markdown summary that preserves:
- Key questions the user asked
- Tickers and instruments discussed
- Important findings, signals, and convictions
- Any unresolved questions or follow-ups

Use short bullet points grouped under headings. Do NOT add commentary beyond
what is in the conversation. Output Markdown only."""


@dataclass
class CompactionResult:
    """Result of a compaction operation."""

    summary: str
    turn_count: int
    input_tokens: int
    output_tokens: int
    cost: float
    compacted_at: datetime = field(default_factory=datetime.now)


class SessionCompactor:
    """Compresses session turns into a Markdown summary via the reporter LLM.

    Usage::

        compactor = SessionCompactor(llm_registry)
        result = await compactor.compact(session_logger)
        result = await compactor.compact_and_save(session_logger, output_dir)
    """

    def __init__(
        self,
        llm_registry: LLMRegistry,
        token_threshold: int = DEFAULT_TOKEN_THRESHOLD,
    ) -> None:
        self._llm_registry = llm_registry
        self._token_threshold = token_threshold

    @property
    def token_threshold(self) -> int:
        return self._token_threshold

    def needs_compaction(self, session_logger: SessionLogger) -> bool:
        """Check whether the session has exceeded the token threshold."""
        return session_logger.token_estimate() >= self._token_threshold

    async def compact(self, session_logger: SessionLogger) -> CompactionResult:
        """Compress session turns into a Markdown summary.

        Raises:
            KeyError: If no reporter provider is registered.
            ValueError: If the session log is empty.
        """
        turns = session_logger.read_all()
        if not turns:
            raise ValueError("Cannot compact an empty session log")

        conversation_text = self._turns_to_text(turns)
        provider = self._llm_registry.get(Role.REPORTER)

        request = CompletionRequest(
            messages=[
                Message(role="system", content=_SYSTEM_PROMPT),
                Message(role="user", content=conversation_text),
            ],
            max_tokens=2048,
            temperature=0.0,
        )
        response = await provider.complete(request)

        return CompactionResult(
            summary=response.content,
            turn_count=len(turns),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=response.cost,
        )

    async def compact_and_save(
        self,
        session_logger: SessionLogger,
        output_dir: Path,
    ) -> CompactionResult:
        """Compact and write the Markdown summary to disk.

        The summary file is named after the session JSONL file with a ``.md``
        extension, stored in *output_dir*.
        """
        result = await self.compact(session_logger)
        output_dir.mkdir(parents=True, exist_ok=True)
        md_name = session_logger.path.stem + ".md"
        md_path = output_dir / md_name
        md_path.write_text(result.summary, encoding="utf-8")
        logger.info("Compacted %d turns → %s", result.turn_count, md_path)
        return result

    @staticmethod
    def _turns_to_text(turns: list[TurnRecord]) -> str:
        """Format turn records into plain text for the LLM."""
        lines: list[str] = []
        for t in turns:
            prefix = f"[Turn {t.turn}] {t.role}"
            if t.tool:
                prefix += f" ({t.tool})"
            lines.append(f"{prefix}: {t.content}")
        return "\n".join(lines)
