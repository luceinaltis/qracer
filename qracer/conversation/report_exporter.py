"""ReportExporter — saves analysis results to Markdown, JSON, and PDF files.

Reports are stored in ``~/.qracer/reports/`` with filenames based on
the primary ticker and date.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from qracer.conversation.analysis_loop import AnalysisResult
from qracer.conversation.intent import Intent

logger = logging.getLogger(__name__)


class ReportExporter:
    """Exports analysis results to Markdown, JSON, and/or PDF files.

    Usage::

        exporter = ReportExporter(Path("~/.qracer/reports"))
        path = exporter.save_markdown(intent, analysis, response_text)
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _build_filename(self, intent: Intent, ext: str) -> Path:
        """Build a filename like AAPL-2026-04-06.md from the intent."""
        ticker = intent.tickers[0] if intent.tickers else "general"
        today = datetime.now().strftime("%Y-%m-%d")
        return self._output_dir / f"{ticker}-{today}{ext}"

    def save_markdown(
        self,
        intent: Intent,
        analysis: AnalysisResult,
        response_text: str,
    ) -> Path:
        """Save the analysis as a Markdown report.

        Returns the path to the saved file.
        """
        ticker = intent.tickers[0] if intent.tickers else "general"
        today = datetime.now().strftime("%Y-%m-%d")

        lines: list[str] = [
            f"# Analysis Report: {ticker}",
            f"**Date:** {today}",
            f"**Query:** {intent.raw_query}",
            f"**Intent:** {intent.intent_type.value}",
            f"**Confidence:** {analysis.confidence:.2f}",
            "",
            "---",
            "",
            "## Response",
            "",
            response_text,
            "",
        ]

        # Trade thesis section
        if analysis.trade_thesis is not None:
            t = analysis.trade_thesis
            lines.extend(
                [
                    "---",
                    "",
                    "## Trade Thesis",
                    "",
                    f"- **Ticker:** {t.ticker}",
                    f"- **Entry Zone:** ${t.entry_zone[0]:.2f} – ${t.entry_zone[1]:.2f}",
                    f"- **Target Price:** ${t.target_price:.2f}",
                    f"- **Stop Loss:** ${t.stop_loss:.2f}",
                    f"- **Risk/Reward:** {t.risk_reward_ratio:.2f}",
                    f"- **Conviction:** {t.conviction}/10",
                    f"- **Catalyst:** {t.catalyst}",
                ]
            )
            if t.catalyst_date:
                lines.append(f"- **Catalyst Date:** {t.catalyst_date}")
            lines.extend(["", t.summary, ""])

        # Data sources used
        tools_used = [r.tool for r in analysis.results if r.success]
        if tools_used:
            lines.extend(
                [
                    "---",
                    "",
                    "## Data Sources",
                    "",
                    *[f"- {tool}" for tool in tools_used],
                    "",
                ]
            )

        path = self._build_filename(intent, ".md")
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Report saved: %s", path)
        return path

    def save_json(
        self,
        intent: Intent,
        analysis: AnalysisResult,
        response_text: str,
    ) -> Path:
        """Save the analysis as a JSON file.

        Returns the path to the saved file.
        """
        ticker = intent.tickers[0] if intent.tickers else "general"
        today = datetime.now().strftime("%Y-%m-%d")

        data: dict = {
            "ticker": ticker,
            "date": today,
            "query": intent.raw_query,
            "intent": intent.intent_type.value,
            "confidence": analysis.confidence,
            "iterations": analysis.iterations,
            "response": response_text,
            "tools": [
                {
                    "tool": r.tool,
                    "success": r.success,
                    "source": r.source,
                    "data": r.data,
                }
                for r in analysis.results
            ],
        }

        if analysis.trade_thesis is not None:
            t = analysis.trade_thesis
            data["trade_thesis"] = {
                "ticker": t.ticker,
                "entry_zone": list(t.entry_zone),
                "target_price": t.target_price,
                "stop_loss": t.stop_loss,
                "risk_reward_ratio": t.risk_reward_ratio,
                "catalyst": t.catalyst,
                "catalyst_date": t.catalyst_date,
                "conviction": t.conviction,
                "summary": t.summary,
            }

        path = self._build_filename(intent, ".json")
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info("Report saved: %s", path)
        return path

    def save_pdf(
        self,
        intent: Intent,
        analysis: AnalysisResult,
        response_text: str,
    ) -> Path:
        """Save the analysis as a formatted PDF report.

        Requires the optional ``fpdf2`` dependency — install it with
        ``pip install qracer[pdf]`` (or ``uv sync --extra pdf``).

        Returns the path to the saved file.
        """
        try:
            from fpdf import FPDF
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "PDF export requires the optional 'fpdf2' dependency. "
                "Install it with: pip install 'qracer[pdf]'"
            ) from exc

        ticker = intent.tickers[0] if intent.tickers else "general"
        today = datetime.now().strftime("%Y-%m-%d")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Header
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, f"qracer Analysis: {ticker}", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(
            0,
            6,
            f"Generated: {today} | Confidence: {analysis.confidence:.2f}",
            new_x="LMARGIN",
            new_y="NEXT",
            align="C",
        )
        pdf.ln(6)

        # Query metadata
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Query", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, intent.raw_query)
        pdf.ln(2)

        # Response body
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Response", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, response_text)

        # Trade thesis section (if present)
        if analysis.trade_thesis is not None:
            t = analysis.trade_thesis
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 10, "Trade Thesis", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            thesis_lines = [
                f"Ticker: {t.ticker}",
                f"Entry Zone: ${t.entry_zone[0]:.2f} - ${t.entry_zone[1]:.2f}",
                f"Target: ${t.target_price:.2f} | Stop: ${t.stop_loss:.2f}",
                f"Risk/Reward: {t.risk_reward_ratio:.2f}x",
                f"Conviction: {t.conviction}/10",
                f"Catalyst: {t.catalyst}",
            ]
            if t.catalyst_date:
                thesis_lines.append(f"Catalyst Date: {t.catalyst_date}")
            thesis_lines.extend(["", t.summary])
            pdf.multi_cell(0, 6, "\n".join(thesis_lines))

        # Data sources (successful tools only, matching Markdown output)
        tools_used = [r for r in analysis.results if r.success]
        if tools_used:
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 10, "Data Sources", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
            for r in tools_used:
                pdf.cell(
                    0,
                    5,
                    f"  - [{r.tool}] {r.source}",
                    new_x="LMARGIN",
                    new_y="NEXT",
                )

        path = self._build_filename(intent, ".pdf")
        pdf.output(str(path))
        logger.info("Report saved: %s", path)
        return path
