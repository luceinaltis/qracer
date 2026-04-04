"""Tests for SessionCompactor (Tier 2)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tracer.llm.providers import CompletionResponse, Role
from tracer.llm.registry import LLMRegistry
from tracer.memory.session_compactor import SessionCompactor
from tracer.memory.session_logger import SessionLogger, TurnRecord


@pytest.fixture
def mock_registry() -> LLMRegistry:
    registry = LLMRegistry()
    provider = AsyncMock()
    provider.complete.return_value = CompletionResponse(
        content="# Session Summary\n\n- User asked about AAPL spike\n- Earnings beat expectations",
        model="claude-haiku-4-20250514",
        input_tokens=200,
        output_tokens=50,
        cost=0.0001,
    )
    registry.register("mock", provider, [Role.REPORTER])
    return registry


@pytest.fixture
def session_logger(tmp_path: Path) -> SessionLogger:
    logger = SessionLogger(tmp_path / "test.jsonl")
    logger.append(TurnRecord(turn=1, role="user", content="Why did AAPL spike?"))
    logger.append(
        TurnRecord(turn=1, role="tool_call", content="Fetching news", tool="fetch_news")
    )
    logger.append(
        TurnRecord(turn=1, role="tool_result", content="Earnings beat", success=True)
    )
    logger.append(
        TurnRecord(turn=1, role="assistant", content="AAPL spiked due to earnings.", conviction=8)
    )
    return logger


class TestSessionCompactor:
    async def test_compact_returns_summary(
        self, mock_registry: LLMRegistry, session_logger: SessionLogger
    ) -> None:
        compactor = SessionCompactor(mock_registry)
        result = await compactor.compact(session_logger)

        assert "Session Summary" in result.summary
        assert result.turn_count == 4
        assert result.input_tokens == 200
        assert result.output_tokens == 50
        assert result.cost == 0.0001

    async def test_compact_empty_raises(self, mock_registry: LLMRegistry, tmp_path: Path) -> None:
        empty_logger = SessionLogger(tmp_path / "empty.jsonl")
        compactor = SessionCompactor(mock_registry)

        with pytest.raises(ValueError, match="empty session log"):
            await compactor.compact(empty_logger)

    async def test_compact_and_save(
        self,
        mock_registry: LLMRegistry,
        session_logger: SessionLogger,
        tmp_path: Path,
    ) -> None:
        compactor = SessionCompactor(mock_registry)
        output_dir = tmp_path / "summaries"
        result = await compactor.compact_and_save(session_logger, output_dir)

        md_path = output_dir / "test.md"
        assert md_path.exists()
        assert md_path.read_text(encoding="utf-8") == result.summary

    async def test_needs_compaction(
        self, mock_registry: LLMRegistry, tmp_path: Path
    ) -> None:
        compactor = SessionCompactor(mock_registry, token_threshold=10)
        logger = SessionLogger(tmp_path / "small.jsonl")
        assert not compactor.needs_compaction(logger)

        # Add enough content to exceed threshold
        logger.append(TurnRecord(turn=1, role="user", content="x" * 200))
        assert compactor.needs_compaction(logger)

    async def test_llm_receives_formatted_turns(
        self, mock_registry: LLMRegistry, session_logger: SessionLogger
    ) -> None:
        compactor = SessionCompactor(mock_registry)
        await compactor.compact(session_logger)

        provider = mock_registry.get(Role.REPORTER)
        call_args = provider.complete.call_args[0][0]
        user_msg = call_args.messages[1].content
        assert "[Turn 1] user:" in user_msg
        assert "[Turn 1] tool_call (fetch_news):" in user_msg

    def test_token_threshold_property(self, mock_registry: LLMRegistry) -> None:
        compactor = SessionCompactor(mock_registry, token_threshold=5000)
        assert compactor.token_threshold == 5000
