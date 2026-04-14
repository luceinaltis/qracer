"""FactStore — structured fact persistence for cross-session memory.

Manages trade theses (Phase 1), findings, and session digests in DuckDB.
Follows the same connection pattern as ``storage/repositories.py``.

The store uses its own DuckDB file (``fact_store.duckdb``), separate from
``memory_index.duckdb``, so the existing :class:`MemorySearcher` is
completely unaffected.

SessionDigest persistence complements the free-text Markdown summary
produced by :class:`SessionCompactor` by preserving structured metadata
(tickers discussed, intent types used, thesis ids, turn count) that the
Markdown summary would otherwise lose.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

from qracer.memory.fact_models import PersistedThesis, SessionDigest, ThesisStatus
from qracer.models.base import TradeThesis

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """\
CREATE SEQUENCE IF NOT EXISTS thesis_id_seq START 1;

CREATE TABLE IF NOT EXISTS theses (
    id                INTEGER PRIMARY KEY DEFAULT nextval('thesis_id_seq'),
    ticker            VARCHAR NOT NULL,
    entry_zone_low    DOUBLE NOT NULL,
    entry_zone_high   DOUBLE NOT NULL,
    target_price      DOUBLE NOT NULL,
    stop_loss         DOUBLE NOT NULL,
    risk_reward_ratio DOUBLE NOT NULL,
    catalyst          VARCHAR NOT NULL,
    catalyst_date     VARCHAR,
    conviction        INTEGER NOT NULL,
    summary           VARCHAR NOT NULL,
    status            VARCHAR NOT NULL DEFAULT 'open',
    session_id        VARCHAR NOT NULL,
    created_at        TIMESTAMP NOT NULL,
    updated_at        TIMESTAMP NOT NULL,
    superseded_by     INTEGER
);

CREATE TABLE IF NOT EXISTS session_digests (
    session_id        VARCHAR PRIMARY KEY,
    tickers_discussed VARCHAR[] NOT NULL,
    intent_types_used VARCHAR[] NOT NULL,
    thesis_ids        INTEGER[] NOT NULL,
    key_conclusions   VARCHAR NOT NULL,
    turn_count        INTEGER NOT NULL,
    created_at        TIMESTAMP NOT NULL
);
"""


def _parse_catalyst_date(raw: str | None) -> datetime | None:
    """Best-effort parse of catalyst_date strings into a datetime.

    Handles ISO dates (``2026-05-01``), quarter notation (``Q2 2026``),
    and year-month (``2026-05``).  Returns ``None`` for unparseable values.
    """
    if not raw:
        return None
    raw = raw.strip()
    # ISO date: 2026-05-01
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        pass
    # Quarter: Q1/Q2/Q3/Q4 2026
    m = re.match(r"Q([1-4])\s*(\d{4})", raw, re.IGNORECASE)
    if m:
        quarter, year = int(m.group(1)), int(m.group(2))
        month = (quarter - 1) * 3 + 1
        return datetime(year, month, 1)
    # Year-month: 2026-05
    m = re.match(r"(\d{4})-(\d{2})$", raw)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), 1)
    return None


def _row_to_thesis(row: tuple) -> PersistedThesis:
    """Convert a DuckDB row tuple to a PersistedThesis."""
    return PersistedThesis(
        id=row[0],
        ticker=row[1],
        entry_zone_low=row[2],
        entry_zone_high=row[3],
        target_price=row[4],
        stop_loss=row[5],
        risk_reward_ratio=row[6],
        catalyst=row[7],
        catalyst_date=row[8],
        conviction=row[9],
        summary=row[10],
        status=ThesisStatus(row[11]),
        session_id=row[12],
        created_at=row[13],
        updated_at=row[14],
        superseded_by=row[15],
    )


_SELECT_COLUMNS = (
    "id, ticker, entry_zone_low, entry_zone_high, target_price, stop_loss, "
    "risk_reward_ratio, catalyst, catalyst_date, conviction, summary, "
    "status, session_id, created_at, updated_at, superseded_by"
)


def _row_to_digest(row: tuple) -> SessionDigest:
    """Convert a DuckDB row tuple to a SessionDigest."""
    return SessionDigest(
        session_id=row[0],
        tickers_discussed=list(row[1]) if row[1] is not None else [],
        intent_types_used=list(row[2]) if row[2] is not None else [],
        thesis_ids=list(row[3]) if row[3] is not None else [],
        key_conclusions=row[4],
        turn_count=row[5],
        created_at=row[6],
    )


class FactStore:
    """Structured fact storage for cross-session memory.

    Usage::

        store = FactStore(Path("~/.qracer/fact_store.duckdb"))
        thesis_id = store.save_thesis(trade_thesis, session_id="abc123")
        open_theses = store.get_open_theses(["AAPL"])
        store.close()
    """

    def __init__(self, path: str | Path | None = None) -> None:
        db_path = str(path) if path else ":memory:"
        self._conn = duckdb.connect(db_path)
        self._conn.execute(_SCHEMA_SQL)

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    # ------------------------------------------------------------------
    # Thesis CRUD
    # ------------------------------------------------------------------

    def save_thesis(self, thesis: TradeThesis, session_id: str) -> int:
        """Persist a TradeThesis with automatic supersession handling.

        If an open thesis exists for the same ticker, it is marked as
        ``superseded`` and linked to the new thesis via ``superseded_by``.

        Returns the new thesis id.
        """
        now = datetime.now()

        # Insert the new thesis first to get its id.
        self._conn.execute(
            """
            INSERT INTO theses (
                ticker, entry_zone_low, entry_zone_high, target_price,
                stop_loss, risk_reward_ratio, catalyst, catalyst_date,
                conviction, summary, status, session_id,
                created_at, updated_at, superseded_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                thesis.ticker,
                thesis.entry_zone[0],
                thesis.entry_zone[1],
                thesis.target_price,
                thesis.stop_loss,
                thesis.risk_reward_ratio,
                thesis.catalyst,
                thesis.catalyst_date,
                thesis.conviction,
                thesis.summary,
                ThesisStatus.OPEN.value,
                session_id,
                now,
                now,
                None,  # superseded_by
            ],
        )

        new_id: int = self._conn.execute("SELECT currval('thesis_id_seq')").fetchone()[0]  # type: ignore[index]

        # Supersede any prior open theses on the same ticker.
        self._conn.execute(
            """
            UPDATE theses
            SET status = ?, superseded_by = ?, updated_at = ?
            WHERE ticker = ? AND status = ? AND id != ?
            """,
            [
                ThesisStatus.SUPERSEDED.value,
                new_id,
                now,
                thesis.ticker,
                ThesisStatus.OPEN.value,
                new_id,
            ],
        )

        return new_id

    def get_open_theses(self, tickers: list[str] | None = None) -> list[PersistedThesis]:
        """Get all open theses, optionally filtered by ticker list."""
        if tickers:
            placeholders = ", ".join("?" for _ in tickers)
            rows = self._conn.execute(
                f"SELECT {_SELECT_COLUMNS} FROM theses "
                f"WHERE status = 'open' AND ticker IN ({placeholders}) "
                "ORDER BY created_at DESC",
                tickers,
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT {_SELECT_COLUMNS} FROM theses "
                "WHERE status = 'open' ORDER BY created_at DESC",
            ).fetchall()
        return [_row_to_thesis(r) for r in rows]

    def get_upcoming_catalysts(self, days_ahead: int = 14) -> list[PersistedThesis]:
        """Get open theses with catalyst_date within *days_ahead* of today.

        Best-effort date parsing: ISO dates, quarter notation (``Q2 2026``),
        and year-month (``2026-05``) are supported.  Unparseable dates are
        excluded.
        """
        all_open = self.get_open_theses()
        cutoff = datetime.now() + timedelta(days=days_ahead)
        now = datetime.now()
        result: list[PersistedThesis] = []
        for t in all_open:
            dt = _parse_catalyst_date(t.catalyst_date)
            if dt is not None and now <= dt <= cutoff:
                result.append(t)
        return result

    def get_thesis_history(self, ticker: str, limit: int = 10) -> list[PersistedThesis]:
        """Get all theses for a ticker (all statuses), most recent first."""
        rows = self._conn.execute(
            f"SELECT {_SELECT_COLUMNS} FROM theses "
            "WHERE ticker = ? ORDER BY created_at DESC LIMIT ?",
            [ticker, limit],
        ).fetchall()
        return [_row_to_thesis(r) for r in rows]

    def update_thesis_status(
        self,
        thesis_id: int,
        status: ThesisStatus,
        *,
        superseded_by: int | None = None,
    ) -> None:
        """Update a thesis status (close, invalidate, supersede)."""
        self._conn.execute(
            "UPDATE theses SET status = ?, superseded_by = ?, updated_at = ? WHERE id = ?",
            [status.value, superseded_by, datetime.now(), thesis_id],
        )

    # ------------------------------------------------------------------
    # SessionDigest CRUD
    # ------------------------------------------------------------------

    def save_digest(self, digest: SessionDigest) -> None:
        """Persist a :class:`SessionDigest` (upsert by ``session_id``).

        Subsequent calls for the same ``session_id`` overwrite the prior
        digest, so the table always reflects the latest cumulative state
        of the session (which may be compacted multiple times).
        """
        self._conn.execute(
            "DELETE FROM session_digests WHERE session_id = ?",
            [digest.session_id],
        )
        self._conn.execute(
            """
            INSERT INTO session_digests (
                session_id, tickers_discussed, intent_types_used,
                thesis_ids, key_conclusions, turn_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                digest.session_id,
                list(digest.tickers_discussed),
                list(digest.intent_types_used),
                list(digest.thesis_ids),
                digest.key_conclusions,
                digest.turn_count,
                digest.created_at,
            ],
        )

    def get_sessions_for_ticker(self, ticker: str, limit: int = 20) -> list[SessionDigest]:
        """Return session digests whose ``tickers_discussed`` contains *ticker*.

        Results are ordered by ``created_at`` descending (most recent first).
        """
        rows = self._conn.execute(
            """
            SELECT session_id, tickers_discussed, intent_types_used,
                   thesis_ids, key_conclusions, turn_count, created_at
            FROM session_digests
            WHERE list_contains(tickers_discussed, ?)
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [ticker, limit],
        ).fetchall()
        return [_row_to_digest(r) for r in rows]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> FactStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
