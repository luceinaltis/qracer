"""DuckDB storage backend for QRacer."""

from __future__ import annotations

from pathlib import Path

import duckdb

DEFAULT_DB_PATH = "qracer.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS prices (
    ticker   VARCHAR NOT NULL,
    date     DATE NOT NULL,
    open     DOUBLE NOT NULL,
    high     DOUBLE NOT NULL,
    low      DOUBLE NOT NULL,
    close    DOUBLE NOT NULL,
    volume   BIGINT NOT NULL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY DEFAULT nextval('signal_id_seq'),
    ticker          VARCHAR NOT NULL,
    direction       VARCHAR NOT NULL,
    conviction      DOUBLE NOT NULL,
    thesis          VARCHAR NOT NULL,
    evidence        VARCHAR[],
    contrarian_angle VARCHAR,
    risk_factors    VARCHAR[],
    time_horizon_days INTEGER,
    generated_at    TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    id                 INTEGER PRIMARY KEY DEFAULT nextval('report_id_seq'),
    title              VARCHAR NOT NULL,
    ticker             VARCHAR,
    conviction         DOUBLE NOT NULL,
    what_happened      VARCHAR NOT NULL,
    evidence_chain     VARCHAR[],
    adversarial_check  VARCHAR[],
    verdict            VARCHAR NOT NULL,
    generated_at       TIMESTAMP NOT NULL
);
"""


class TracerDB:
    """Manages the DuckDB connection and schema for Tracer.

    Usage::

        db = TracerDB()          # in-memory
        db = TracerDB("qracer.db")  # file-backed
        conn = db.connection
        db.close()
    """

    def __init__(self, path: str | Path | None = None) -> None:
        db_path = str(path) if path else ":memory:"
        self._conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("CREATE SEQUENCE IF NOT EXISTS signal_id_seq START 1")
        self._conn.execute("CREATE SEQUENCE IF NOT EXISTS report_id_seq START 1")
        self._conn.execute(_SCHEMA_SQL)

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> TracerDB:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
