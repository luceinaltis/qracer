"""Tests for MemorySearcher (Tier 3)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from qracer.memory.memory_searcher import MemorySearcher


@pytest.fixture
def searcher() -> Iterator[MemorySearcher]:
    s = MemorySearcher()
    try:
        # Pre-warm the FTS extension so tests that rely on keyword search
        # skip cleanly when the extension cannot be installed (e.g. offline).
        s._ensure_fts_loaded()
    except Exception:
        s.close()
        pytest.skip("DuckDB FTS extension unavailable")
    yield s
    s.close()


class TestMemorySearcher:
    def test_index_and_search(self, searcher: MemorySearcher) -> None:
        searcher.index_summary("sess_001", "AAPL earnings beat expectations, stock spiked 8%")
        searcher.index_summary("sess_002", "Fed rate decision, inflation trending down")

        results = searcher.search("AAPL earnings")
        assert len(results) >= 1
        assert results[0].session_id == "sess_001"
        assert results[0].score > 0

    def test_search_empty_index(self, searcher: MemorySearcher) -> None:
        results = searcher.search("anything")
        assert results == []

    def test_search_no_match(self, searcher: MemorySearcher) -> None:
        searcher.index_summary("sess_001", "AAPL earnings analysis")
        results = searcher.search("cryptocurrency bitcoin")
        # FTS may return low-relevance results or none; just verify no crash
        assert isinstance(results, list)

    def test_index_replaces_on_conflict(self, searcher: MemorySearcher) -> None:
        searcher.index_summary("sess_001", "Old summary about AAPL")
        searcher.index_summary("sess_001", "New summary about MSFT and Microsoft")

        results = searcher.search("MSFT Microsoft")
        assert len(results) >= 1
        assert "MSFT" in results[0].summary

    def test_remove(self, searcher: MemorySearcher) -> None:
        searcher.index_summary("sess_001", "AAPL analysis")
        searcher.remove("sess_001")

        # After removal, the table should be empty
        row = searcher.connection.execute("SELECT count(*) FROM session_index").fetchone()
        assert row is not None and row[0] == 0

    def test_index_directory(self, searcher: MemorySearcher, tmp_path: Path) -> None:
        (tmp_path / "sess_001.md").write_text("# AAPL\nEarnings beat", encoding="utf-8")
        (tmp_path / "sess_002.md").write_text("# MSFT\nCloud revenue up", encoding="utf-8")
        (tmp_path / "empty.md").write_text("", encoding="utf-8")
        (tmp_path / "not_md.txt").write_text("Ignored", encoding="utf-8")

        count = searcher.index_directory(tmp_path)
        assert count == 2  # empty.md skipped, .txt ignored

    def test_search_limit(self, searcher: MemorySearcher) -> None:
        for i in range(5):
            searcher.index_summary(f"sess_{i:03d}", f"Analysis of market trends session {i}")

        results = searcher.search("market trends", limit=2)
        assert len(results) <= 2

    def test_context_manager(self) -> None:
        try:
            s = MemorySearcher()
            s._ensure_fts_loaded()
        except Exception:
            pytest.skip("DuckDB FTS extension unavailable")
        s.close()

        with MemorySearcher() as s:
            s.index_summary("sess_001", "Test content")
            results = s.search("Test")
            assert len(results) >= 1

    def test_search_result_fields(self, searcher: MemorySearcher) -> None:
        searcher.index_summary("sess_001", "AAPL earnings beat expectations")
        results = searcher.search("AAPL earnings")
        assert len(results) == 1

        r = results[0]
        assert r.session_id == "sess_001"
        assert r.summary == "AAPL earnings beat expectations"
        assert isinstance(r.score, float)
        assert r.indexed_at is not None


# ---------------------------------------------------------------------------
# Hybrid (keyword + vector) search
# ---------------------------------------------------------------------------


def _fake_embedding(text: str) -> list[float]:
    """Deterministic 3-dim embedding keyed on which topic words appear.

    Used by tests to exercise vector search without pulling in a real
    embedding model. The axes represent: AAPL/earnings, Fed/rates, crypto.
    """
    text_lower = text.lower()
    dims = [
        1.0 if any(k in text_lower for k in ("aapl", "earnings")) else 0.0,
        1.0 if any(k in text_lower for k in ("fed", "rate", "inflation")) else 0.0,
        1.0 if any(k in text_lower for k in ("crypto", "bitcoin")) else 0.0,
    ]
    # Avoid all-zero vectors so cosine similarity is defined.
    if sum(dims) == 0:
        dims[0] = 0.01
    return dims


@pytest.fixture
def hybrid_searcher() -> Iterator[MemorySearcher]:
    """Hybrid searcher whose tests do not require the FTS extension.

    Vector-only paths do not depend on FTS, so we avoid pre-warming it here
    and only skip if DuckDB itself fails to open.
    """
    s = MemorySearcher(embedding_fn=_fake_embedding)
    yield s
    s.close()


class TestHybridSearch:
    def test_has_embeddings_flag(self, hybrid_searcher: MemorySearcher) -> None:
        assert hybrid_searcher.has_embeddings is True

    def test_kw_searcher_has_no_embeddings(self, searcher: MemorySearcher) -> None:
        assert searcher.has_embeddings is False

    def test_index_summary_stores_embedding(self, hybrid_searcher: MemorySearcher) -> None:
        hybrid_searcher.index_summary("sess_001", "AAPL earnings were strong")
        row = hybrid_searcher.connection.execute(
            "SELECT embedding FROM session_embeddings WHERE session_id = 'sess_001'"
        ).fetchone()
        assert row is not None
        assert list(row[0]) == _fake_embedding("AAPL earnings were strong")

    def test_vector_search_finds_semantic_match(self, hybrid_searcher: MemorySearcher) -> None:
        hybrid_searcher.index_summary("sess_aapl", "Quarterly earnings for the iPhone maker")
        hybrid_searcher.index_summary("sess_fed", "Fed held rates steady amid sticky inflation")
        hybrid_searcher.index_summary(
            "sess_btc", "Crypto market rallied as bitcoin broke resistance"
        )

        # Query shares the Fed/rates axis but not exact keywords.
        results = hybrid_searcher._vector_search("rate decision outlook", limit=5)
        assert results, "vector search should return at least one hit"
        assert results[0].session_id == "sess_fed"

    def test_search_is_hybrid_when_embedding_fn_set(self, hybrid_searcher: MemorySearcher) -> None:
        hybrid_searcher.index_summary("sess_aapl", "AAPL earnings beat expectations")
        hybrid_searcher.index_summary("sess_fed", "Fed policy and inflation trajectory")

        results = hybrid_searcher.search("AAPL earnings", limit=5)
        assert results
        session_ids = [r.session_id for r in results]
        assert "sess_aapl" in session_ids
        # RRF scores are always positive, and both branches contributed.
        assert all(r.score > 0 for r in results)

    def test_merge_results_reciprocal_rank_fusion(self, hybrid_searcher: MemorySearcher) -> None:
        from datetime import datetime

        from qracer.memory.memory_searcher import SearchResult

        now = datetime.now()
        keyword = [
            SearchResult("sess_a", "a", 5.0, now),
            SearchResult("sess_b", "b", 3.0, now),
        ]
        vector = [
            SearchResult("sess_b", "b", 0.95, now),
            SearchResult("sess_c", "c", 0.80, now),
        ]

        merged = hybrid_searcher._merge_results(keyword, vector, limit=10)

        ids = [r.session_id for r in merged]
        # sess_b appears in both lists → should rank first under RRF.
        assert ids[0] == "sess_b"
        assert set(ids) == {"sess_a", "sess_b", "sess_c"}
        # Scores must be strictly decreasing.
        scores = [r.score for r in merged]
        assert scores == sorted(scores, reverse=True)

    def test_remove_clears_embedding_row(self, hybrid_searcher: MemorySearcher) -> None:
        hybrid_searcher.index_summary("sess_001", "AAPL earnings")
        hybrid_searcher.remove("sess_001")
        row = hybrid_searcher.connection.execute(
            "SELECT count(*) FROM session_embeddings"
        ).fetchone()
        assert row is not None and row[0] == 0

    def test_vector_search_returns_empty_without_embedding_fn(
        self, searcher: MemorySearcher
    ) -> None:
        searcher.index_summary("sess_001", "AAPL earnings")
        assert searcher._vector_search("AAPL", limit=5) == []

    def test_embedding_failure_does_not_block_indexing(self) -> None:
        def broken(_: str) -> list[float]:
            raise RuntimeError("embedding service down")

        with MemorySearcher(embedding_fn=broken) as s:
            # Indexing must still succeed even though the embedding fn
            # throws — the summary row is written before embedding is
            # attempted.
            s.index_summary("sess_001", "AAPL earnings beat")

            row = s.connection.execute(
                "SELECT count(*) FROM session_index WHERE session_id = 'sess_001'"
            ).fetchone()
            assert row is not None and row[0] == 1

            # The embedding row should be absent because the fn raised.
            emb_row = s.connection.execute(
                "SELECT count(*) FROM session_embeddings WHERE session_id = 'sess_001'"
            ).fetchone()
            assert emb_row is not None and emb_row[0] == 0
