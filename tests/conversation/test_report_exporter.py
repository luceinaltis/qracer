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


# fpdf2 is an optional dependency; skip this whole class if it isn't installed.
fpdf = pytest.importorskip("fpdf")


def _read_pdf_text(path) -> str:
    """Extract text from a PDF written by fpdf2.

    fpdf2 wraps content streams in zlib (FlateDecode) by default, so we
    walk the raw bytes, pull each ``stream ... endstream`` block, try to
    decompress it, and grep for ``(string) Tj`` and ``[(string) ...] TJ``
    text-showing operators. This avoids adding a PDF parser dependency
    just for tests.
    """
    import re
    import zlib

    raw = path.read_bytes()
    # PDF spec: a stream object is bracketed by the literal keywords
    # `stream` and `endstream`, separated by EOL markers.
    streams = re.findall(rb"stream\r?\n(.*?)\r?\nendstream", raw, re.DOTALL)
    parts: list[str] = []
    for chunk in streams:
        try:
            decoded = zlib.decompress(chunk)
        except zlib.error:
            decoded = chunk
        text = decoded.decode("latin-1", errors="replace")
        parts.extend(re.findall(r"\(((?:\\.|[^()])*)\)\s*Tj", text))
        for arr in re.findall(r"\[(.*?)\]\s*TJ", text):
            parts.extend(re.findall(r"\(((?:\\.|[^()])*)\)", arr))
    return "\n".join(parts)


class TestReportExporterPdf:
    def test_save_basic(self, tmp_path) -> None:
        exporter = ReportExporter(tmp_path)
        path = exporter.save_pdf(_intent(), _analysis(), "Analysis text here.")
        assert path.exists()
        assert path.suffix == ".pdf"
        # PDFs always start with the %PDF- magic bytes.
        assert path.read_bytes().startswith(b"%PDF-")

    def test_save_includes_header_and_response(self, tmp_path) -> None:
        exporter = ReportExporter(tmp_path)
        path = exporter.save_pdf(_intent(), _analysis(), "Analysis text here.")
        text = _read_pdf_text(path)
        assert "AAPL" in text
        assert "Analysis text here." in text
        assert "0.85" in text
        assert "Response" in text

    def test_save_with_thesis(self, tmp_path) -> None:
        exporter = ReportExporter(tmp_path)
        path = exporter.save_pdf(_intent(), _analysis(with_thesis=True), "Response body")
        text = _read_pdf_text(path)
        assert "Trade Thesis" in text
        assert "$175.00" in text
        assert "$200.00" in text
        assert "8/10" in text
        assert "AI revenue growth" in text
        assert "Q2 2026" in text

    def test_save_with_data_sources_status(self, tmp_path) -> None:
        exporter = ReportExporter(tmp_path)
        path = exporter.save_pdf(_intent(), _analysis(), "Response")
        text = _read_pdf_text(path)
        assert "Data Sources" in text
        assert "price_event" in text
        assert "news" in text
        # Failed tools must be reported with a FAIL marker, per #138 spec.
        assert "macro" in text
        assert "[FAIL]" in text
        assert "[OK]" in text

    def test_general_ticker_fallback(self, tmp_path) -> None:
        exporter = ReportExporter(tmp_path)
        intent = Intent(intent_type=IntentType.MACRO_QUERY, tickers=[], raw_query="inflation?")
        path = exporter.save_pdf(intent, _analysis(), "Response")
        assert path.suffix == ".pdf"
        assert "general" in path.name

    def test_handles_unicode_in_response(self, tmp_path) -> None:
        """Non-latin-1 characters must not crash PDF generation."""
        exporter = ReportExporter(tmp_path)
        path = exporter.save_pdf(
            _intent(),
            _analysis(),
            "Earnings beat — strong demand in 한국 (Korea) 📈",
        )
        assert path.exists()
        assert path.read_bytes().startswith(b"%PDF-")
