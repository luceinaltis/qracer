"""Structured fact models for cross-session memory persistence.

These models represent structured knowledge extracted from analysis sessions
(trade theses, findings, session digests) that survive across REPL sessions.
They map to DuckDB tables managed by :class:`FactStore`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ThesisStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"


@dataclass
class PersistedThesis:
    """A trade thesis persisted to the fact store.

    Maps to the runtime ``TradeThesis`` from ``models/base.py`` with added
    lifecycle metadata (id, status, session_id, timestamps, superseded_by).
    """

    id: int
    ticker: str
    entry_zone_low: float
    entry_zone_high: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    catalyst: str
    catalyst_date: str | None
    conviction: int  # 1-10
    summary: str
    status: ThesisStatus
    session_id: str
    created_at: datetime
    updated_at: datetime
    superseded_by: int | None = None


@dataclass
class Finding:
    """A discrete analytical insight extracted from analysis."""

    id: int
    entity: str  # ticker or macro indicator name
    statement: str
    confidence: float  # 0.0-1.0
    source_tool: str
    session_id: str
    event_date: str | None = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SessionDigest:
    """Lightweight structured summary of a session."""

    session_id: str
    tickers_discussed: list[str]
    intent_types_used: list[str]
    thesis_ids: list[int]
    key_conclusions: str
    turn_count: int
    created_at: datetime = field(default_factory=datetime.now)
