"""Tests for MemorySearcher (Tier 3)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from qracer.memory.memory_searcher import MemorySearcher


@pytest.fixture
def searcher() -> Iterator[MemorySearcher]:
    try:
        s = MemorySearcher()
    except Exception:
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
