"""Tests for TracerDB."""

from __future__ import annotations

import pytest

from tracer.storage.database import TracerDB


class TestTracerDB:
    def test_creates_tables(self) -> None:
        with TracerDB() as db:
            tables = db.connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()
            table_names = [t[0] for t in tables]
            assert "prices" in table_names
            assert "signals" in table_names
            assert "reports" in table_names

    def test_schema_idempotent(self) -> None:
        with TracerDB() as db:
            db._init_schema()  # noqa: SLF001
            tables = db.connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
            assert len([t for t in tables if t[0] == "prices"]) == 1

    def test_context_manager(self) -> None:
        with TracerDB() as db:
            conn = db.connection
            conn.execute("SELECT 1")
        with pytest.raises(Exception):
            conn.execute("SELECT 1")

    def test_file_backed(self, tmp_path: object) -> None:
        db_file = str(tmp_path) + "/test.db"  # type: ignore[operator]
        with TracerDB(db_file) as db:
            db.connection.execute(
                "INSERT INTO prices VALUES ('AAPL', '2024-01-01', 1, 2, 0.5, 1.5, 100)"
            )
        with TracerDB(db_file) as db:
            rows = db.connection.execute("SELECT * FROM prices").fetchall()
            assert len(rows) == 1
