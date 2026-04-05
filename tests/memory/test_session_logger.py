"""Tests for SessionLogger (Tier 1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from qracer.memory.session_logger import SessionLogger, TurnRecord


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "test_session.jsonl"


@pytest.fixture
def logger(log_path: Path) -> SessionLogger:
    return SessionLogger(log_path)


class TestTurnRecord:
    def test_to_json_drops_none_fields(self) -> None:
        record = TurnRecord(turn=1, role="user", content="Hello")
        raw = record.to_json()
        assert '"tool"' not in raw
        assert '"args"' not in raw
        assert '"turn": 1' in raw

    def test_roundtrip(self) -> None:
        record = TurnRecord(
            turn=2,
            role="tool_call",
            content="Fetching news",
            tool="fetch_news",
            args={"ticker": "AAPL"},
        )
        restored = TurnRecord.from_json(record.to_json())
        assert restored.turn == 2
        assert restored.role == "tool_call"
        assert restored.tool == "fetch_news"
        assert restored.args == {"ticker": "AAPL"}

    def test_tool_result_fields(self) -> None:
        record = TurnRecord(
            turn=3,
            role="tool_result",
            content="Success",
            success=True,
            source="Finnhub",
        )
        restored = TurnRecord.from_json(record.to_json())
        assert restored.success is True
        assert restored.source == "Finnhub"

    def test_conviction_field(self) -> None:
        record = TurnRecord(turn=1, role="assistant", content="Analysis", conviction=8.5)
        restored = TurnRecord.from_json(record.to_json())
        assert restored.conviction == 8.5


class TestSessionLogger:
    def test_append_and_read(self, logger: SessionLogger) -> None:
        logger.append(TurnRecord(turn=1, role="user", content="Why did AAPL spike?"))
        logger.append(TurnRecord(turn=1, role="assistant", content="Earnings beat."))

        turns = logger.read_all()
        assert len(turns) == 2
        assert turns[0].role == "user"
        assert turns[1].content == "Earnings beat."

    def test_read_empty(self, log_path: Path) -> None:
        logger = SessionLogger(log_path)
        assert logger.read_all() == []

    def test_read_nonexistent(self, tmp_path: Path) -> None:
        logger = SessionLogger(tmp_path / "does_not_exist.jsonl")
        assert logger.read_all() == []

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "session.jsonl"
        logger = SessionLogger(nested)
        logger.append(TurnRecord(turn=1, role="user", content="test"))
        assert nested.exists()

    def test_turn_count(self, logger: SessionLogger) -> None:
        assert logger.turn_count() == 0
        logger.append(TurnRecord(turn=1, role="user", content="hi"))
        logger.append(TurnRecord(turn=1, role="assistant", content="hello"))
        assert logger.turn_count() == 2

    def test_token_estimate(self, logger: SessionLogger) -> None:
        assert logger.token_estimate() == 0
        logger.append(TurnRecord(turn=1, role="user", content="x" * 400))
        # ~400 chars of content + JSON overhead → should be > 100 tokens
        assert logger.token_estimate() > 100

    def test_path_property(self, logger: SessionLogger, log_path: Path) -> None:
        assert logger.path == log_path

    def test_append_preserves_order(self, logger: SessionLogger) -> None:
        for i in range(5):
            logger.append(TurnRecord(turn=i, role="user", content=f"msg {i}"))
        turns = logger.read_all()
        assert [t.turn for t in turns] == [0, 1, 2, 3, 4]
