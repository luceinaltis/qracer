"""Tests for ReportExporter."""

from __future__ import annotations

import json

import pytest

from qracer.conversation.analysis_loop import AnalysisResult
from qracer.conversation.intent import Intent, IntentType
from qracer.conversation.report_exporter import ReportExporter
from qracer.models import ToolResult, TradeThesis


def _intent(tickers: list[str] | None = None) -> Intent:
    return Intent(
        intent_type=IntentType.DEEP_DIVE,
        tickers=tickers or ["AAPL"],
        raw_query="Analyze AAPL",
    )


def _analysis(with_thesis: bool = False) -> AnalysisResult:
    results = [
        ToolResult(tool="price_event", success=True, data={"price": 180.0}, source="test"),
        ToolResult(tool="news", success=True, data={"count": 3}, source="test"),
        ToolResult(tool="macro", success=False, data={}, source="test", error="no provider"),
    ]
    thesis = None
    if with_thesis:
        thesis = TradeThesis(
            ticker="AAPL",
            entry_zone=(175.0, 180.0),
            target_price=200.0,
            stop_loss=165.0,
            risk_reward_ratio=2.5,
            catalyst="AI revenue growth",
            catalyst_date="Q2 2026",
            conviction=8,
            summary="Long AAPL on AI tailwinds.",
        )
    return AnalysisResult(results=results, confidence=0.85, iterations=2, trade_thesis=thesis)


class TestReportExporterMarkdown:
    def test_save_basic(self, tmp_path) -> None:
        exporter = ReportExporter(tmp_path)
        path = exporter.save_markdown(_intent(), _analysis(), "Analysis text here.")
        assert path.exists()
        assert path.suffix == ".md"
        content = path.read_text()
        assert "AAPL" in content
        assert "Analysis text here." in content
        assert "0.85" in content

    def test_save_with_thesis(self, tmp_path) -> None:
        exporter = ReportExporter(tmp_path)
        path = exporter.save_markdown(_intent(), _analysis(with_thesis=True), "Response")
        content = path.read_text()
        assert "Trade Thesis" in content
        assert "$175.00" in content
        assert "$200.00" in content
        assert "8/10" in content
        assert "AI revenue growth" in content
        assert "Q2 2026" in content

    def test_save_with_data_sources(self, tmp_path) -> None:
        exporter = ReportExporter(tmp_path)
        path = exporter.save_markdown(_intent(), _analysis(), "Response")
        content = path.read_text()
        assert "Data Sources" in content
        assert "price_event" in content
        assert "news" in content
        # Failed tool (macro) should not appear in data sources
        assert "macro" not in content.split("Data Sources")[1]

    def test_general_ticker_fallback(self, tmp_path) -> None:
        exporter = ReportExporter(tmp_path)
        intent = Intent(intent_type=IntentType.MACRO_QUERY, tickers=[], raw_query="inflation?")
        path = exporter.save_markdown(intent, _analysis(), "Response")
        assert "general" in path.name


class TestReportExporterJson:
    def test_save_basic(self, tmp_path) -> None:
        exporter = ReportExporter(tmp_path)
        path = exporter.save_json(_intent(), _analysis(), "Analysis text.")
        assert path.exists()
        assert path.suffix == ".json"
        data = json.loads(path.read_text())
        assert data["ticker"] == "AAPL"
        assert data["confidence"] == 0.85
        assert data["response"] == "Analysis text."
        assert len(data["tools"]) == 3

    def test_save_with_thesis(self, tmp_path) -> None:
        exporter = ReportExporter(tmp_path)
        path = exporter.save_json(_intent(), _analysis(with_thesis=True), "Response")
        data = json.loads(path.read_text())
        assert "trade_thesis" in data
        assert data["trade_thesis"]["conviction"] == 8
        assert data["trade_thesis"]["target_price"] == 200.0

    def test_no_thesis_key_when_none(self, tmp_path) -> None:
        exporter = ReportExporter(tmp_path)
        path = exporter.save_json(_intent(), _analysis(with_thesis=False), "Response")
        data = json.loads(path.read_text())
        assert "trade_thesis" not in data


class TestReportExporterPdf:
    """PDF export requires the optional ``fpdf2`` dependency."""

    def test_save_basic(self, tmp_path) -> None:
        pytest.importorskip("fpdf")
        exporter = ReportExporter(tmp_path)
        path = exporter.save_pdf(_intent(), _analysis(), "Analysis text here.")
        assert path.exists()
        assert path.suffix == ".pdf"
        # PDF files start with the %PDF- magic header.
        assert path.read_bytes().startswith(b"%PDF-")
        assert path.stat().st_size > 0

    def test_save_with_thesis(self, tmp_path) -> None:
        pytest.importorskip("fpdf")
        exporter = ReportExporter(tmp_path)
        path = exporter.save_pdf(_intent(), _analysis(with_thesis=True), "Response")
        assert path.exists()
        # fpdf2 compresses streams, so we only assert that the file is larger
        # than a minimal "no thesis" report — a rough sanity check.
        baseline = exporter.save_pdf(_intent(tickers=["MSFT"]), _analysis(), "Response")
        assert path.stat().st_size > baseline.stat().st_size

    def test_general_ticker_fallback(self, tmp_path) -> None:
        pytest.importorskip("fpdf")
        exporter = ReportExporter(tmp_path)
        intent = Intent(intent_type=IntentType.MACRO_QUERY, tickers=[], raw_query="inflation?")
        path = exporter.save_pdf(intent, _analysis(), "Response")
        assert "general" in path.name
        assert path.suffix == ".pdf"

    def test_unicode_response_sanitized(self, tmp_path) -> None:
        """Smart quotes and em dashes must not crash fpdf2's Latin-1 encoder."""
        pytest.importorskip("fpdf")
        exporter = ReportExporter(tmp_path)
        unicode_text = "AAPL \u2014 strong \u201cbuy\u201d signal \u2026 bullish \u2022 growth"
        path = exporter.save_pdf(_intent(), _analysis(), unicode_text)
        assert path.exists()
        assert path.read_bytes().startswith(b"%PDF-")

    def test_raises_when_fpdf_missing(self, tmp_path, monkeypatch) -> None:
        """save_pdf should raise a helpful ImportError if fpdf2 is not installed."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "fpdf":
                raise ImportError("No module named 'fpdf'")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        exporter = ReportExporter(tmp_path)
        with pytest.raises(ImportError, match="fpdf2"):
            exporter.save_pdf(_intent(), _analysis(), "Response")
