"""SessionLogger — append-only JSONL audit log (Tier 1).

Every conversation turn is written as a single JSON line: user messages,
tool calls, tool results, and assistant responses.  The log is the immutable
source of truth for what happened in a session.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnRecord:
    """A single entry in the session audit log."""

    turn: int
    role: str  # "user", "assistant", "tool_call", "tool_result"
    content: str
    ts: str = field(default_factory=lambda: datetime.now().isoformat())
    tool: str | None = None
    args: dict[str, Any] | None = None
    success: bool | None = None
    source: str | None = None
    conviction: float | None = None

    def to_json(self) -> str:
        """Serialise to a compact JSON string, dropping None fields."""
        d = {k: v for k, v in asdict(self).items() if v is not None}
        return json.dumps(d, default=str)

    @classmethod
    def from_json(cls, line: str) -> TurnRecord:
        """Deserialise a JSON line back into a TurnRecord."""
        d = json.loads(line)
        return cls(**d)


class SessionLogger:
    """Append-only JSONL logger for a single conversation session.

    Usage::

        log = SessionLogger(Path("sessions/abc123.jsonl"))
        log.append(TurnRecord(turn=1, role="user", content="Why did AAPL spike?"))
        turns = log.read_all()
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: TurnRecord) -> None:
        """Append a turn record to the JSONL file."""
        with self._path.open("a", encoding="utf-8") as f:
            f.write(record.to_json() + "\n")

    def read_all(self) -> list[TurnRecord]:
        """Read every turn record from the log."""
        if not self._path.exists():
            return []
        records: list[TurnRecord] = []
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(TurnRecord.from_json(line))
        return records

    def token_estimate(self) -> int:
        """Rough token count (~4 chars per token) for compaction threshold."""
        if not self._path.exists():
            return 0
        text = self._path.read_text(encoding="utf-8")
        return len(text) // 4

    def turn_count(self) -> int:
        """Return the number of records in the log."""
        return len(self.read_all())
