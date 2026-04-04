"""Repository classes for DuckDB storage."""

from __future__ import annotations

from datetime import date

import duckdb

from tracer.data.providers import OHLCV
from tracer.models.base import Report, Signal, SignalDirection


class PriceRepository:
    """Append-only OHLCV price storage."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def upsert(self, ticker: str, bars: list[OHLCV]) -> int:
        """Insert or replace OHLCV bars. Returns number of rows affected."""
        if not bars:
            return 0
        data = [
            (ticker, b.date, b.open, b.high, b.low, b.close, b.volume) for b in bars
        ]
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO prices (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            data,
        )
        return len(data)

    def get(self, ticker: str, start: date, end: date) -> list[OHLCV]:
        """Retrieve OHLCV bars for a ticker within a date range."""
        rows = self._conn.execute(
            """
            SELECT date, open, high, low, close, volume
            FROM prices
            WHERE ticker = ? AND date BETWEEN ? AND ?
            ORDER BY date
            """,
            [ticker, start, end],
        ).fetchall()
        return [
            OHLCV(date=r[0], open=r[1], high=r[2], low=r[3], close=r[4], volume=r[5])
            for r in rows
        ]

    def latest_date(self, ticker: str) -> date | None:
        """Return the most recent date stored for a ticker, or None."""
        row = self._conn.execute(
            "SELECT MAX(date) FROM prices WHERE ticker = ?", [ticker]
        ).fetchone()
        return row[0] if row and row[0] is not None else None


class SignalRepository:
    """Signal persistence."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def save(self, signal: Signal) -> int:
        """Persist a signal. Returns the generated id."""
        result = self._conn.execute(
            """
            INSERT INTO signals
                (ticker, direction, conviction, thesis, evidence,
                 contrarian_angle, risk_factors, time_horizon_days, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            [
                signal.ticker,
                signal.direction.value,
                signal.conviction,
                signal.thesis,
                signal.evidence,
                signal.contrarian_angle,
                signal.risk_factors,
                signal.time_horizon_days,
                signal.generated_at,
            ],
        ).fetchone()
        return result[0]  # type: ignore[index]

    def get_by_ticker(self, ticker: str, limit: int = 50) -> list[Signal]:
        """Retrieve signals for a ticker, most recent first."""
        rows = self._conn.execute(
            """
            SELECT ticker, direction, conviction, thesis, evidence,
                   contrarian_angle, risk_factors, time_horizon_days, generated_at
            FROM signals
            WHERE ticker = ?
            ORDER BY generated_at DESC
            LIMIT ?
            """,
            [ticker, limit],
        ).fetchall()
        return [self._row_to_signal(r) for r in rows]

    def get_recent(self, limit: int = 50) -> list[Signal]:
        """Retrieve the most recent signals across all tickers."""
        rows = self._conn.execute(
            """
            SELECT ticker, direction, conviction, thesis, evidence,
                   contrarian_angle, risk_factors, time_horizon_days, generated_at
            FROM signals
            ORDER BY generated_at DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        return [self._row_to_signal(r) for r in rows]

    @staticmethod
    def _row_to_signal(r: tuple) -> Signal:  # type: ignore[type-arg]
        return Signal(
            ticker=r[0],
            direction=SignalDirection(r[1]),
            conviction=r[2],
            thesis=r[3],
            evidence=list(r[4]) if r[4] else [],
            contrarian_angle=r[5],
            risk_factors=list(r[6]) if r[6] else [],
            time_horizon_days=r[7],
            generated_at=r[8],
        )


class ReportRepository:
    """Report persistence."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def save(self, report: Report) -> int:
        """Persist a report. Returns the generated id."""
        result = self._conn.execute(
            """
            INSERT INTO reports
                (title, ticker, conviction, what_happened,
                 evidence_chain, adversarial_check, verdict, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            [
                report.title,
                report.ticker,
                report.conviction,
                report.what_happened,
                report.evidence_chain,
                report.adversarial_check,
                report.verdict,
                report.generated_at,
            ],
        ).fetchone()
        return result[0]  # type: ignore[index]

    def get_by_ticker(self, ticker: str, limit: int = 20) -> list[Report]:
        """Retrieve reports for a ticker, most recent first."""
        rows = self._conn.execute(
            """
            SELECT title, ticker, conviction, what_happened,
                   evidence_chain, adversarial_check, verdict, generated_at
            FROM reports
            WHERE ticker = ?
            ORDER BY generated_at DESC
            LIMIT ?
            """,
            [ticker, limit],
        ).fetchall()
        return [self._row_to_report(r) for r in rows]

    def get_recent(self, limit: int = 20) -> list[Report]:
        """Retrieve the most recent reports across all tickers."""
        rows = self._conn.execute(
            """
            SELECT title, ticker, conviction, what_happened,
                   evidence_chain, adversarial_check, verdict, generated_at
            FROM reports
            ORDER BY generated_at DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        return [self._row_to_report(r) for r in rows]

    @staticmethod
    def _row_to_report(r: tuple) -> Report:  # type: ignore[type-arg]
        return Report(
            title=r[0],
            ticker=r[1],
            conviction=r[2],
            what_happened=r[3],
            evidence_chain=list(r[4]) if r[4] else [],
            adversarial_check=list(r[5]) if r[5] else [],
            verdict=r[6],
            generated_at=r[7],
        )
