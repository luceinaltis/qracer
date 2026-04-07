"""Tests for ReportExporter — Markdown, JSON, and PDF export."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from tracer.conversation.engine import AnalysisResult, EngineResponse
from tracer.conversation.intent import Intent, IntentType
from tracer.conversation.report_exporter import ReportExporter
from tracer.models import ToolResult, TradeThesis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_response(*, with_thesis: bool = True) -> EngineResponse:
    """Build a minimal EngineResponse for testing."""
    intent = Intent(
        intent_type=IntentType.DEEP_DIVE,
        tickers=["AAPL"],
        tools=["price_event", "fundamentals"],
        raw_query="deep dive AAPL",
    )

    tool_results = [
        ToolResult(tool="price_event", success=True, data={"price": 185.0}, source="yfinance"),
        ToolResult(
            tool="fundamentals",
            success=False,
            data={},
            source="yfinance",
            error="timeout",
        ),
    ]

    thesis = None
    if with_thesis:
        thesis = TradeThesis(
            ticker="AAPL",
            entry_zone=(180.0, 185.0),
            target_price=210.0,
            stop_loss=170.0,
            risk_reward_ratio=2.0,
            catalyst="Q2 earnings beat",
            catalyst_date="2026-07-01",
            conviction=7,
            summary="AAPL looks strong heading into earnings.",
        )

    analysis = AnalysisResult(
        results=tool_results,
        confidence=0.8,
        iterations=2,
        trade_thesis=thesis,
    )

    return EngineResponse(
        text="AAPL analysis response text.",
        intent=intent,
        analysis=analysis,
        generated_at=datetime(2026, 4, 7, 12, 0, 0),
    )


# ---------------------------------------------------------------------------
# Markdown tests
# ---------------------------------------------------------------------------


class TestSaveMarkdown:
    def test_creates_md_file(self, tmp_path: Path) -> None:
        exporter = ReportExporter(output_dir=tmp_path)
        path = exporter.save_markdown(_make_response())

        assert path.exists()
        assert path.suffix == ".md"
        assert "AAPL" in path.name

    def test_content_includes_key_sections(self, tmp_path: Path) -> None:
        exporter = ReportExporter(output_dir=tmp_path)
        path = exporter.save_markdown(_make_response())
        content = path.read_text()

        assert "# Analysis: AAPL" in content
        assert "Trade Thesis" in content
        assert "$180.00" in content
        assert "Data Sources" in content
        assert "FAILED (timeout)" in content

    def test_without_thesis(self, tmp_path: Path) -> None:
        exporter = ReportExporter(output_dir=tmp_path)
        path = exporter.save_markdown(_make_response(with_thesis=False))
        content = path.read_text()

        assert "Trade Thesis" not in content


# ---------------------------------------------------------------------------
# JSON tests
# ---------------------------------------------------------------------------


class TestSaveJson:
    def test_creates_json_file(self, tmp_path: Path) -> None:
        exporter = ReportExporter(output_dir=tmp_path)
        path = exporter.save_json(_make_response())

        assert path.exists()
        assert path.suffix == ".json"

    def test_valid_json_structure(self, tmp_path: Path) -> None:
        exporter = ReportExporter(output_dir=tmp_path)
        path = exporter.save_json(_make_response())
        data = json.loads(path.read_text())

        assert data["title"] == "Analysis: AAPL"
        assert data["intent"]["tickers"] == ["AAPL"]
        assert data["analysis"]["confidence"] == 0.8
        assert data["analysis"]["trade_thesis"]["conviction"] == 7
        assert len(data["data_sources"]) == 2

    def test_without_thesis(self, tmp_path: Path) -> None:
        exporter = ReportExporter(output_dir=tmp_path)
        path = exporter.save_json(_make_response(with_thesis=False))
        data = json.loads(path.read_text())

        assert data["analysis"]["trade_thesis"] is None


# ---------------------------------------------------------------------------
# PDF tests
# ---------------------------------------------------------------------------


class TestSavePdf:
    def test_creates_pdf_file(self, tmp_path: Path) -> None:
        exporter = ReportExporter(output_dir=tmp_path)
        path = exporter.save_pdf(_make_response())

        assert path.exists()
        assert path.suffix == ".pdf"
        # PDF files start with %PDF
        raw = path.read_bytes()
        assert raw[:5] == b"%PDF-"

    def test_pdf_without_thesis(self, tmp_path: Path) -> None:
        exporter = ReportExporter(output_dir=tmp_path)
        path = exporter.save_pdf(_make_response(with_thesis=False))

        assert path.exists()
        raw = path.read_bytes()
        assert raw[:5] == b"%PDF-"

    def test_pdf_nonzero_size(self, tmp_path: Path) -> None:
        exporter = ReportExporter(output_dir=tmp_path)
        path = exporter.save_pdf(_make_response())

        assert path.stat().st_size > 500  # a real PDF with content

    def test_pdf_import_error_without_fpdf2(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify a clear error when fpdf2 is not installed."""
        import builtins

        real_import = builtins.__import__

        def _mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "fpdf":
                raise ModuleNotFoundError("No module named 'fpdf'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        exporter = ReportExporter(output_dir=tmp_path)
        with pytest.raises(RuntimeError, match="fpdf2"):
            exporter.save_pdf(_make_response())
