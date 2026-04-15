"""Tests for MEMORY.md / BOOTSTRAP.md long-term memory helpers."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from qracer.memory.fact_store import FactStore
from qracer.memory.memory_file import (
    AUTO_BEGIN,
    AUTO_END,
    MemoryDocument,
    load_bootstrap,
    load_memory,
    parse_memory,
    refresh_memory,
    refresh_memory_file,
    render_memory,
    save_memory,
)
from qracer.models.base import TradeThesis


def _thesis(
    ticker: str = "AAPL",
    catalyst_date: str | None = None,
    conviction: int = 8,
) -> TradeThesis:
    return TradeThesis(
        ticker=ticker,
        entry_zone=(175.0, 180.0),
        target_price=200.0,
        stop_loss=165.0,
        risk_reward_ratio=2.5,
        catalyst="AI revenue growth",
        catalyst_date=catalyst_date,
        conviction=conviction,
        summary=f"Long {ticker} on AI tailwinds",
    )


@pytest.fixture
def fact_store() -> Iterator[FactStore]:
    store = FactStore()  # in-memory
    try:
        yield store
    finally:
        store.close()


# ----------------------------------------------------------------------
# Render / parse
# ----------------------------------------------------------------------


class TestRenderMemory:
    def test_empty_doc_renders_placeholders(self) -> None:
        text = render_memory(MemoryDocument())
        assert AUTO_BEGIN in text
        assert AUTO_END in text
        assert "## Active Theses" in text
        assert "_No open theses yet._" in text
        assert "_No upcoming catalysts within the horizon._" in text

    def test_theses_and_catalysts_rendered_as_bullets(self) -> None:
        doc = MemoryDocument(
            auto_theses=["**AAPL** (conviction 8/10): ..."],
            auto_catalysts=["AAPL: AI revenue growth — Q2 2026"],
        )
        text = render_memory(doc)
        assert "- **AAPL** (conviction 8/10): ..." in text
        assert "- AAPL: AI revenue growth — Q2 2026" in text

    def test_user_content_appears_after_auto_region(self) -> None:
        doc = MemoryDocument(user_content="## Notes\n\nHand-curated thoughts.")
        text = render_memory(doc)
        auto_idx = text.index(AUTO_END)
        user_idx = text.index("Hand-curated thoughts.")
        assert user_idx > auto_idx


class TestParseMemory:
    def test_roundtrip_preserves_all_fields(self) -> None:
        original = MemoryDocument(
            auto_theses=["**AAPL** (conviction 8/10): summary."],
            auto_catalysts=["AAPL: earnings — Q2 2026"],
            user_content="## Watchpoints\n\nFed meeting.",
            last_updated=datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc),
        )
        parsed = parse_memory(render_memory(original))
        assert parsed.auto_theses == original.auto_theses
        assert parsed.auto_catalysts == original.auto_catalysts
        assert "Fed meeting." in parsed.user_content
        assert parsed.last_updated == original.last_updated

    def test_empty_text_yields_default(self) -> None:
        doc = parse_memory("")
        assert doc.auto_theses == []
        assert doc.auto_catalysts == []
        assert "Watchpoints" in doc.user_content

    def test_missing_auto_region_preserves_user_body(self) -> None:
        text = "# qracer MEMORY.md\n\n## My Notes\n\nArbitrary text.\n"
        doc = parse_memory(text)
        assert doc.auto_theses == []
        assert "Arbitrary text." in doc.user_content
        # Header should not bleed into user content.
        assert "# qracer MEMORY.md" not in doc.user_content

    def test_malformed_timestamp_falls_back_to_now(self) -> None:
        text = (
            "# qracer MEMORY.md\n\n*Last updated: not-a-date*\n\n"
            f"{AUTO_BEGIN}\n## Active Theses\n\n{AUTO_END}\n"
        )
        before = datetime.now(timezone.utc)
        doc = parse_memory(text)
        after = datetime.now(timezone.utc)
        assert before <= doc.last_updated <= after

    def test_bullets_outside_theses_heading_ignored(self) -> None:
        text = (
            f"{AUTO_BEGIN}\n"
            "## Active Theses\n\n- a\n- b\n\n"
            "## Upcoming Catalysts\n\n- x\n\n"
            f"{AUTO_END}\n"
        )
        doc = parse_memory(text)
        assert doc.auto_theses == ["a", "b"]
        assert doc.auto_catalysts == ["x"]

    def test_user_content_between_auto_and_trailing_sections(self) -> None:
        text = (
            "# qracer MEMORY.md\n\n"
            f"{AUTO_BEGIN}\n## Active Theses\n\n- a\n\n## Upcoming Catalysts\n\n{AUTO_END}\n\n"
            "## Watchpoints\n\nFed May.\n"
        )
        doc = parse_memory(text)
        assert doc.auto_theses == ["a"]
        assert "Fed May." in doc.user_content


# ----------------------------------------------------------------------
# Refresh from FactStore
# ----------------------------------------------------------------------


class TestRefreshMemory:
    def test_regenerates_auto_region_from_fact_store(self, fact_store: FactStore) -> None:
        fact_store.save_thesis(_thesis("AAPL"), session_id="s1")
        fact_store.save_thesis(_thesis("NVDA", conviction=9), session_id="s1")

        doc = refresh_memory(MemoryDocument(), fact_store)
        assert len(doc.auto_theses) == 2
        tickers_rendered = " ".join(doc.auto_theses)
        assert "**AAPL**" in tickers_rendered
        assert "**NVDA**" in tickers_rendered
        assert "conviction 9/10" in tickers_rendered

    def test_preserves_user_content(self, fact_store: FactStore) -> None:
        fact_store.save_thesis(_thesis("AAPL"), session_id="s1")
        original = MemoryDocument(user_content="## Notes\n\nKeep me.")
        refreshed = refresh_memory(original, fact_store)
        assert refreshed.user_content == "## Notes\n\nKeep me."

    def test_upcoming_catalysts_within_horizon(self, fact_store: FactStore) -> None:
        from datetime import timedelta

        near = (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d")
        fact_store.save_thesis(_thesis("AAPL", catalyst_date=near), session_id="s1")
        doc = refresh_memory(MemoryDocument(), fact_store, catalyst_horizon_days=30)
        assert any("AAPL" in line for line in doc.auto_catalysts)

    def test_catalyst_outside_horizon_excluded(self, fact_store: FactStore) -> None:
        from datetime import timedelta

        far = (datetime.now() + timedelta(days=120)).strftime("%Y-%m-%d")
        fact_store.save_thesis(_thesis("AAPL", catalyst_date=far), session_id="s1")
        doc = refresh_memory(MemoryDocument(), fact_store, catalyst_horizon_days=30)
        assert doc.auto_catalysts == []

    def test_no_theses_yields_empty_auto_lists(self, fact_store: FactStore) -> None:
        doc = refresh_memory(MemoryDocument(), fact_store)
        assert doc.auto_theses == []
        assert doc.auto_catalysts == []

    def test_last_updated_bumped(self, fact_store: FactStore) -> None:
        old = MemoryDocument(last_updated=datetime(2020, 1, 1, tzinfo=timezone.utc))
        refreshed = refresh_memory(old, fact_store)
        assert refreshed.last_updated > old.last_updated


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        doc = MemoryDocument(
            auto_theses=["**AAPL** (conviction 8/10): x."],
            user_content="## Mine\n\nHello.",
        )
        path = tmp_path / "MEMORY.md"
        save_memory(doc, path)
        loaded = load_memory(path)
        assert loaded.auto_theses == doc.auto_theses
        assert "Hello." in loaded.user_content

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "MEMORY.md"
        save_memory(MemoryDocument(), path)
        assert path.exists()

    def test_load_missing_file_returns_default(self, tmp_path: Path) -> None:
        doc = load_memory(tmp_path / "absent.md")
        assert doc.auto_theses == []
        assert "Watchpoints" in doc.user_content

    def test_refresh_memory_file_preserves_user_section(
        self, tmp_path: Path, fact_store: FactStore
    ) -> None:
        path = tmp_path / "MEMORY.md"
        save_memory(MemoryDocument(user_content="## Mine\n\nKeep this."), path)
        fact_store.save_thesis(_thesis("AAPL"), session_id="s1")
        refreshed = refresh_memory_file(path, fact_store)
        assert "Keep this." in refreshed.user_content
        on_disk = path.read_text(encoding="utf-8")
        assert "Keep this." in on_disk
        assert "**AAPL**" in on_disk

    def test_save_is_atomic(self, tmp_path: Path) -> None:
        """``.tmp`` file should not linger after a successful save."""
        path = tmp_path / "MEMORY.md"
        save_memory(MemoryDocument(), path)
        assert not (tmp_path / "MEMORY.md.tmp").exists()


# ----------------------------------------------------------------------
# BOOTSTRAP.md
# ----------------------------------------------------------------------


class TestLoadBootstrap:
    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        assert load_bootstrap(tmp_path / "absent.md") is None

    def test_returns_none_for_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "BOOTSTRAP.md"
        path.write_text("   \n\n", encoding="utf-8")
        assert load_bootstrap(path) is None

    def test_returns_stripped_content(self, tmp_path: Path) -> None:
        path = tmp_path / "BOOTSTRAP.md"
        path.write_text("\nLong-term value investor.\n", encoding="utf-8")
        assert load_bootstrap(path) == "Long-term value investor."


# ----------------------------------------------------------------------
# Summary line
# ----------------------------------------------------------------------


class TestSummaryLine:
    def test_counts_reflected(self) -> None:
        doc = MemoryDocument(auto_theses=["a", "b", "c"], auto_catalysts=["x", "y"])
        assert "3 active theses" in doc.summary_line()
        assert "2 upcoming catalysts" in doc.summary_line()
