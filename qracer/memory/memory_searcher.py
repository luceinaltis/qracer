"""MemorySearcher — hybrid retrieval over session summaries (Tier 3).

Indexes Tier 2 Markdown summaries in DuckDB for keyword search via full-text
search (FTS) and, optionally, semantic similarity via stored embeddings.  The
Markdown files on disk remain the source of truth; DuckDB is the search index
only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

EmbeddingFn = Callable[[str], list[float]]

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS session_index (
    session_id  VARCHAR PRIMARY KEY,
    summary     VARCHAR NOT NULL,
    indexed_at  TIMESTAMP NOT NULL
);
"""

_EMBEDDINGS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS session_embeddings (
    session_id  VARCHAR PRIMARY KEY,
    embedding   FLOAT[] NOT NULL,
    indexed_at  TIMESTAMP NOT NULL
);
"""

_FTS_INDEX_SQL = """
PRAGMA create_fts_index('session_index', 'session_id', 'summary', overwrite=1);
"""

# Reciprocal rank fusion constant (Cormack et al. 2009).
_RRF_K = 60

# Process-wide cache: the DuckDB FTS extension download can take 80+s to time
# out in offline environments, so we avoid repeating the attempt.
_FTS_AVAILABLE: bool | None = None


@dataclass(frozen=True)
class SearchResult:
    """A single search hit from the memory index."""

    session_id: str
    summary: str
    score: float
    indexed_at: datetime


class MemorySearcher:
    """Hybrid keyword + vector search over compacted session summaries.

    Keyword search uses DuckDB FTS (BM25).  Vector search is optional and
    requires an ``embedding_fn`` callable that maps a string to a dense
    vector; when provided, :meth:`search` runs both branches and fuses
    results via reciprocal rank fusion.

    Usage::

        searcher = MemorySearcher()                      # keyword only
        searcher = MemorySearcher(embedding_fn=my_embed) # hybrid search
        searcher.index_summary("sess_001", "# AAPL analysis ...")
        results = searcher.search("AAPL earnings")
        searcher.close()
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        embedding_fn: EmbeddingFn | None = None,
    ) -> None:
        db_path = str(path) if path else ":memory:"
        self._conn = duckdb.connect(db_path)
        self._embedding_fn = embedding_fn
        self._fts_loaded = False
        self._init_schema()
        self._fts_dirty = True

    def _init_schema(self) -> None:
        self._conn.execute(_SCHEMA_SQL)
        self._conn.execute(_EMBEDDINGS_SCHEMA_SQL)

    def _ensure_fts_loaded(self) -> None:
        """Install/load the DuckDB FTS extension on first use.

        Deferred until a keyword search actually runs so that pure
        vector-search workloads (and offline test environments that
        cannot download DuckDB extensions) are not impacted.  Failures
        are cached at process level to avoid repeatedly paying the 80+s
        download timeout.
        """
        global _FTS_AVAILABLE
        if self._fts_loaded:
            return
        if _FTS_AVAILABLE is False:
            raise RuntimeError("DuckDB FTS extension unavailable")
        try:
            self._conn.execute("INSTALL fts")
            self._conn.execute("LOAD fts")
        except Exception:
            _FTS_AVAILABLE = False
            raise
        _FTS_AVAILABLE = True
        self._fts_loaded = True

    def _rebuild_fts(self) -> None:
        """Rebuild the FTS index if data has changed since last build."""
        if not self._fts_dirty:
            return
        self._ensure_fts_loaded()
        count = self._conn.execute("SELECT count(*) FROM session_index").fetchone()
        if count and count[0] > 0:
            self._conn.execute(_FTS_INDEX_SQL)
        self._fts_dirty = False

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    @property
    def has_embeddings(self) -> bool:
        """Whether this searcher was configured with an embedding function."""
        return self._embedding_fn is not None

    def index_summary(self, session_id: str, summary: str) -> None:
        """Insert or replace a session summary in the search index.

        When an embedding function is configured, the summary is also
        embedded and stored in ``session_embeddings``.
        """
        now = datetime.now()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO session_index (session_id, summary, indexed_at)
            VALUES (?, ?, ?)
            """,
            [session_id, summary, now],
        )
        self._fts_dirty = True

        if self._embedding_fn is not None:
            try:
                vector = self._embedding_fn(summary)
            except Exception:
                logger.warning(
                    "Embedding function failed for session %s", session_id, exc_info=True
                )
                return
            self._conn.execute(
                """
                INSERT OR REPLACE INTO session_embeddings (session_id, embedding, indexed_at)
                VALUES (?, ?, ?)
                """,
                [session_id, vector, now],
            )

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
        """Run a keyword (or hybrid) search over indexed summaries.

        Returns results ordered by relevance score (descending).  When an
        embedding function is configured, keyword and vector hits are fused
        with reciprocal rank fusion; otherwise only keyword results are
        returned.
        """
        kw_results = self._keyword_search(query, limit)
        if self._embedding_fn is None:
            return kw_results

        vec_results = self._vector_search(query, limit)
        return self._merge_results(kw_results, vec_results, limit)

    def _keyword_search(self, query: str, limit: int) -> list[SearchResult]:
        """Run BM25 keyword search via DuckDB FTS.

        If the FTS extension cannot be loaded (for instance in an offline
        environment), this branch degrades to an empty result set rather
        than failing the whole hybrid search.
        """
        try:
            self._rebuild_fts()
        except Exception:
            logger.warning("FTS keyword search unavailable", exc_info=True)
            return []

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

    def _vector_search(self, query: str, limit: int) -> list[SearchResult]:
        """Run cosine similarity search using the stored embeddings.

        Returns an empty list when no embedding function is configured, the
        embedding call fails, or the embeddings table is empty.
        """
        if self._embedding_fn is None:
            return []

        row_count = self._conn.execute("SELECT count(*) FROM session_embeddings").fetchone()
        if not row_count or row_count[0] == 0:
            return []

        try:
            query_vector = self._embedding_fn(query)
        except Exception:
            logger.warning("Embedding function failed for query", exc_info=True)
            return []

        rows = self._conn.execute(
            """
            SELECT
                si.session_id,
                si.summary,
                list_cosine_similarity(se.embedding, ?::FLOAT[]) AS score,
                si.indexed_at
            FROM session_embeddings se
            JOIN session_index si USING (session_id)
            WHERE score IS NOT NULL
            ORDER BY score DESC
            LIMIT ?
            """,
            [query_vector, limit],
        ).fetchall()

        return [
            SearchResult(
                session_id=r[0],
                summary=r[1],
                score=float(r[2]),
                indexed_at=r[3],
            )
            for r in rows
        ]

    def _merge_results(
        self,
        keyword_results: list[SearchResult],
        vector_results: list[SearchResult],
        limit: int,
    ) -> list[SearchResult]:
        """Fuse keyword and vector hits with reciprocal rank fusion.

        RRF combines rankings without needing to normalise the underlying
        scores (BM25 and cosine similarity are on different scales).  Each
        document's fused score is ``sum(1 / (k + rank_i))`` across input
        lists that contain it, where *rank* is 1-based.
        """
        scores: dict[str, float] = {}
        records: dict[str, SearchResult] = {}

        for rank, result in enumerate(keyword_results, start=1):
            scores[result.session_id] = scores.get(result.session_id, 0.0) + 1.0 / (_RRF_K + rank)
            records[result.session_id] = result

        for rank, result in enumerate(vector_results, start=1):
            scores[result.session_id] = scores.get(result.session_id, 0.0) + 1.0 / (_RRF_K + rank)
            # Prefer the keyword-side record if present; otherwise use vector.
            records.setdefault(result.session_id, result)

        fused = [
            SearchResult(
                session_id=sid,
                summary=records[sid].summary,
                score=score,
                indexed_at=records[sid].indexed_at,
            )
            for sid, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        ]
        return fused[:limit]

    def remove(self, session_id: str) -> None:
        """Remove a session from the index."""
        self._conn.execute("DELETE FROM session_index WHERE session_id = ?", [session_id])
        self._conn.execute("DELETE FROM session_embeddings WHERE session_id = ?", [session_id])
        self._fts_dirty = True

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> MemorySearcher:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
