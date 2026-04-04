"""MemorySearcher — hybrid retrieval over session summaries (Tier 3).

Indexes Tier 2 Markdown summaries in DuckDB for keyword search via full-text
search (FTS).  The Markdown files on disk remain the source of truth; DuckDB
is the search index only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS session_index (
    session_id  VARCHAR PRIMARY KEY,
    summary     VARCHAR NOT NULL,
    indexed_at  TIMESTAMP NOT NULL
);
"""

_FTS_INDEX_SQL = """
PRAGMA create_fts_index('session_index', 'session_id', 'summary', overwrite=1);
"""


@dataclass(frozen=True)
class SearchResult:
    """A single search hit from the memory index."""

    session_id: str
    summary: str
    score: float
    indexed_at: datetime


class MemorySearcher:
    """Keyword search over compacted session summaries using DuckDB FTS.

    Usage::

        searcher = MemorySearcher()                    # in-memory
        searcher = MemorySearcher("memory_index.db")   # file-backed
        searcher.index_summary("sess_001", "# AAPL analysis ...")
        results = searcher.search("AAPL earnings")
        searcher.close()
    """

    def __init__(self, path: str | Path | None = None) -> None:
        db_path = str(path) if path else ":memory:"
        self._conn = duckdb.connect(db_path)
        self._init_schema()
        self._fts_dirty = True

    def _init_schema(self) -> None:
        self._conn.execute("INSTALL fts")
        self._conn.execute("LOAD fts")
        self._conn.execute(_SCHEMA_SQL)

    def _rebuild_fts(self) -> None:
        """Rebuild the FTS index if data has changed since last build."""
        if not self._fts_dirty:
            return
        count = self._conn.execute("SELECT count(*) FROM session_index").fetchone()
        if count and count[0] > 0:
            self._conn.execute(_FTS_INDEX_SQL)
        self._fts_dirty = False

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    def index_summary(self, session_id: str, summary: str) -> None:
        """Insert or replace a session summary in the search index."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO session_index (session_id, summary, indexed_at)
            VALUES (?, ?, ?)
            """,
            [session_id, summary, datetime.now()],
        )
        self._fts_dirty = True

    def index_directory(self, summaries_dir: Path) -> int:
        """Index all ``.md`` files in a directory. Returns count of files indexed."""
        count = 0
        for md_file in sorted(summaries_dir.glob("*.md")):
            session_id = md_file.stem
            summary = md_file.read_text(encoding="utf-8")
            if summary.strip():
                self.index_summary(session_id, summary)
                count += 1
        return count

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Run a keyword search over indexed summaries.

        Returns results ordered by relevance score (descending).
        """
        self._rebuild_fts()

        row_count = self._conn.execute("SELECT count(*) FROM session_index").fetchone()
        if not row_count or row_count[0] == 0:
            return []

        rows = self._conn.execute(
            """
            SELECT
                session_id,
                summary,
                fts_main_session_index.match_bm25(session_id, ?) AS score,
                indexed_at
            FROM session_index
            WHERE score IS NOT NULL
            ORDER BY score DESC
            LIMIT ?
            """,
            [query, limit],
        ).fetchall()

        return [
            SearchResult(
                session_id=r[0],
                summary=r[1],
                score=r[2],
                indexed_at=r[3],
            )
            for r in rows
        ]

    def remove(self, session_id: str) -> None:
        """Remove a session from the index."""
        self._conn.execute(
            "DELETE FROM session_index WHERE session_id = ?", [session_id]
        )
        self._fts_dirty = True

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> MemorySearcher:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
