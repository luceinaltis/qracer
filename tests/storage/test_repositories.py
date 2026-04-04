"""Tests for repository classes."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from tracer.data.providers import OHLCV
from tracer.models.base import Report, Signal, SignalDirection
from tracer.storage.database import TracerDB
from tracer.storage.repositories import PriceRepository, ReportRepository, SignalRepository


@pytest.fixture
def db() -> TracerDB:
    """In-memory database for each test."""
    _db = TracerDB()
    yield _db  # type: ignore[misc]
    _db.close()


class TestPriceRepository:
    def test_upsert_and_get(self, db: TracerDB) -> None:
        repo = PriceRepository(db.connection)
        bars = [
            OHLCV(date=date(2024, 1, 2), open=100, high=105, low=99, close=103, volume=1000),
            OHLCV(date=date(2024, 1, 3), open=103, high=108, low=102, close=107, volume=1200),
        ]
        count = repo.upsert("AAPL", bars)
        assert count == 2

        result = repo.get("AAPL", date(2024, 1, 1), date(2024, 1, 5))
        assert len(result) == 2
        assert result[0].date == date(2024, 1, 2)
        assert result[1].close == 107

    def test_upsert_replaces_on_conflict(self, db: TracerDB) -> None:
        repo = PriceRepository(db.connection)
        bar = OHLCV(date=date(2024, 1, 2), open=100, high=105, low=99, close=103, volume=1000)
        repo.upsert("AAPL", [bar])

        updated = OHLCV(date=date(2024, 1, 2), open=100, high=110, low=99, close=109, volume=1500)
        repo.upsert("AAPL", [updated])

        result = repo.get("AAPL", date(2024, 1, 1), date(2024, 1, 5))
        assert len(result) == 1
        assert result[0].close == 109

    def test_upsert_empty(self, db: TracerDB) -> None:
        repo = PriceRepository(db.connection)
        assert repo.upsert("AAPL", []) == 0

    def test_get_empty(self, db: TracerDB) -> None:
        repo = PriceRepository(db.connection)
        assert repo.get("AAPL", date(2024, 1, 1), date(2024, 1, 5)) == []

    def test_latest_date(self, db: TracerDB) -> None:
        repo = PriceRepository(db.connection)
        assert repo.latest_date("AAPL") is None

        bars = [
            OHLCV(date=date(2024, 1, 2), open=1, high=2, low=0.5, close=1.5, volume=100),
            OHLCV(date=date(2024, 1, 5), open=1, high=2, low=0.5, close=1.5, volume=100),
        ]
        repo.upsert("AAPL", bars)
        assert repo.latest_date("AAPL") == date(2024, 1, 5)

    def test_ticker_isolation(self, db: TracerDB) -> None:
        repo = PriceRepository(db.connection)
        bar_a = OHLCV(date=date(2024, 1, 2), open=1, high=2, low=0.5, close=1.5, volume=100)
        bar_b = OHLCV(date=date(2024, 1, 2), open=10, high=20, low=5, close=15, volume=200)
        repo.upsert("AAPL", [bar_a])
        repo.upsert("MSFT", [bar_b])

        result = repo.get("AAPL", date(2024, 1, 1), date(2024, 1, 5))
        assert len(result) == 1
        assert result[0].close == 1.5


class TestSignalRepository:
    def _make_signal(self, ticker: str = "AAPL", **kwargs: object) -> Signal:
        defaults: dict = {
            "ticker": ticker,
            "direction": SignalDirection.LONG,
            "conviction": 8.0,
            "thesis": "Strong momentum",
            "evidence": ["Revenue up", "Guidance raised"],
            "risk_factors": ["Valuation stretched"],
            "generated_at": datetime(2024, 6, 15, 10, 0),
        }
        defaults.update(kwargs)
        return Signal(**defaults)

    def test_save_and_get(self, db: TracerDB) -> None:
        repo = SignalRepository(db.connection)
        signal = self._make_signal()
        signal_id = repo.save(signal)
        assert isinstance(signal_id, int)

        results = repo.get_by_ticker("AAPL")
        assert len(results) == 1
        s = results[0]
        assert s.ticker == "AAPL"
        assert s.direction == SignalDirection.LONG
        assert s.conviction == 8.0
        assert s.evidence == ["Revenue up", "Guidance raised"]
        assert s.risk_factors == ["Valuation stretched"]

    def test_get_recent(self, db: TracerDB) -> None:
        repo = SignalRepository(db.connection)
        repo.save(self._make_signal("AAPL", generated_at=datetime(2024, 6, 1)))
        repo.save(self._make_signal("MSFT", generated_at=datetime(2024, 6, 2)))
        repo.save(self._make_signal("GOOG", generated_at=datetime(2024, 6, 3)))

        results = repo.get_recent(limit=2)
        assert len(results) == 2
        assert results[0].ticker == "GOOG"
        assert results[1].ticker == "MSFT"

    def test_optional_fields(self, db: TracerDB) -> None:
        repo = SignalRepository(db.connection)
        signal = Signal(
            ticker="X",
            direction=SignalDirection.NEUTRAL,
            conviction=5.0,
            thesis="Flat",
            generated_at=datetime(2024, 1, 1),
        )
        repo.save(signal)
        result = repo.get_by_ticker("X")[0]
        assert result.evidence == []
        assert result.contrarian_angle is None
        assert result.risk_factors == []
        assert result.time_horizon_days is None


class TestReportRepository:
    def _make_report(self, ticker: str = "AAPL", **kwargs: object) -> Report:
        defaults: dict = {
            "title": f"{ticker} Analysis",
            "ticker": ticker,
            "conviction": 8.0,
            "what_happened": "Price spike on earnings",
            "evidence_chain": ["Revenue beat", "Margin expansion"],
            "adversarial_check": ["Seasonal effect possible"],
            "verdict": "High conviction long",
            "generated_at": datetime(2024, 6, 15, 10, 0),
        }
        defaults.update(kwargs)
        return Report(**defaults)

    def test_save_and_get(self, db: TracerDB) -> None:
        repo = ReportRepository(db.connection)
        report = self._make_report()
        report_id = repo.save(report)
        assert isinstance(report_id, int)

        results = repo.get_by_ticker("AAPL")
        assert len(results) == 1
        r = results[0]
        assert r.title == "AAPL Analysis"
        assert r.conviction == 8.0
        assert r.evidence_chain == ["Revenue beat", "Margin expansion"]
        assert r.adversarial_check == ["Seasonal effect possible"]

    def test_get_recent(self, db: TracerDB) -> None:
        repo = ReportRepository(db.connection)
        repo.save(self._make_report("AAPL", generated_at=datetime(2024, 6, 1)))
        repo.save(self._make_report("MSFT", generated_at=datetime(2024, 6, 2)))

        results = repo.get_recent(limit=1)
        assert len(results) == 1
        assert results[0].ticker == "MSFT"

    def test_null_ticker(self, db: TracerDB) -> None:
        repo = ReportRepository(db.connection)
        report = Report(
            title="Macro Overview",
            ticker=None,
            conviction=6.0,
            what_happened="Fed rate decision",
            evidence_chain=["CPI trending down"],
            adversarial_check=["Employment still strong"],
            verdict="Cautious outlook",
            generated_at=datetime(2024, 6, 15),
        )
        repo.save(report)
        results = repo.get_recent()
        assert len(results) == 1
        assert results[0].ticker is None
