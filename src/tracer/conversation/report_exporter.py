"""ReportExporter — save analysis results as Markdown, JSON, or PDF."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tracer.conversation.engine import EngineResponse

logger = logging.getLogger(__name__)


class ReportExporter:
    """Export an EngineResponse to various file formats."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self._output_dir = output_dir or Path.cwd()

    def _ensure_dir(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _build_filename(self, response: EngineResponse, ext: str) -> Path:
        tickers = "_".join(response.intent.tickers) if response.intent.tickers else "general"
        ts = response.generated_at.strftime("%Y%m%d_%H%M%S")
        return self._output_dir / f"tracer_{tickers}_{ts}.{ext}"

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def save_markdown(self, response: EngineResponse) -> Path:
        """Save the analysis as a Markdown file."""
        self._ensure_dir()
        path = self._build_filename(response, "md")

        lines: list[str] = []
        ticker_str = ", ".join(response.intent.tickers) if response.intent.tickers else "General"
        ts = response.generated_at.strftime("%Y-%m-%d %H:%M:%S")

        lines.append(f"# Analysis: {ticker_str}")
        lines.append(f"*Generated: {ts}*\n")
        lines.append(response.text)
        lines.append("")

        # Trade thesis section
        thesis = response.analysis.trade_thesis
        if thesis is not None:
            lines.append("## Trade Thesis")
            lines.append(f"- **Entry Zone:** ${thesis.entry_zone[0]:.2f} – ${thesis.entry_zone[1]:.2f}")
            lines.append(f"- **Target Price:** ${thesis.target_price:.2f}")
            lines.append(f"- **Stop Loss:** ${thesis.stop_loss:.2f}")
            lines.append(f"- **Risk/Reward:** {thesis.risk_reward_ratio:.2f}")
            lines.append(f"- **Conviction:** {thesis.conviction}/10")
            lines.append(f"- **Catalyst:** {thesis.catalyst}")
            lines.append(f"\n{thesis.summary}")
            lines.append("")

        # Data sources
        lines.append("## Data Sources")
        for r in response.analysis.results:
            status = "OK" if r.success else f"FAILED ({r.error})"
            lines.append(f"- **{r.tool}** ({r.source}): {status}")
        lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Saved Markdown report to %s", path)
        return path

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def save_json(self, response: EngineResponse) -> Path:
        """Save the analysis as a JSON file."""
        self._ensure_dir()
        path = self._build_filename(response, "json")

        thesis_data = None
        thesis = response.analysis.trade_thesis
        if thesis is not None:
            thesis_data = {
                "ticker": thesis.ticker,
                "entry_zone": list(thesis.entry_zone),
                "target_price": thesis.target_price,
                "stop_loss": thesis.stop_loss,
                "risk_reward_ratio": thesis.risk_reward_ratio,
                "catalyst": thesis.catalyst,
                "catalyst_date": thesis.catalyst_date,
                "conviction": thesis.conviction,
                "summary": thesis.summary,
            }

        payload = {
            "title": f"Analysis: {', '.join(response.intent.tickers) or 'General'}",
            "generated_at": response.generated_at.isoformat(),
            "intent": {
                "type": response.intent.intent_type.value,
                "tickers": response.intent.tickers,
                "tools": response.intent.tools,
                "raw_query": response.intent.raw_query,
            },
            "response_text": response.text,
            "analysis": {
                "confidence": response.analysis.confidence,
                "iterations": response.analysis.iterations,
                "early_exit_reason": response.analysis.early_exit_reason,
                "trade_thesis": thesis_data,
            },
            "data_sources": [
                {
                    "tool": r.tool,
                    "source": r.source,
                    "success": r.success,
                    "error": r.error,
                }
                for r in response.analysis.results
            ],
        }

        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        logger.info("Saved JSON report to %s", path)
        return path

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    def save_pdf(self, response: EngineResponse) -> Path:
        """Save the analysis as a formatted PDF file.

        Requires the optional ``fpdf2`` dependency:
            pip install tracer[pdf]
        """
        try:
            from fpdf import FPDF
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PDF export requires fpdf2. Install it with: pip install tracer[pdf]"
            ) from exc

        self._ensure_dir()
        path = self._build_filename(response, "pdf")

        ticker_str = ", ".join(response.intent.tickers) if response.intent.tickers else "General"
        ts = response.generated_at.strftime("%Y-%m-%d %H:%M:%S")

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # --- Header ---
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, f"Analysis: {ticker_str}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 6, f"Generated: {ts}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)

        # --- Response body ---
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Response", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, response.text)
        pdf.ln(4)

        # --- Trade thesis ---
        thesis = response.analysis.trade_thesis
        if thesis is not None:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Trade Thesis", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)

            rows = [
                ("Entry Zone", f"${thesis.entry_zone[0]:.2f} - ${thesis.entry_zone[1]:.2f}"),
                ("Target Price", f"${thesis.target_price:.2f}"),
                ("Stop Loss", f"${thesis.stop_loss:.2f}"),
                ("Risk/Reward", f"{thesis.risk_reward_ratio:.2f}"),
                ("Conviction", f"{thesis.conviction}/10"),
                ("Catalyst", thesis.catalyst),
            ]
            for label, value in rows:
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(40, 6, f"{label}:")
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")

            pdf.ln(2)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, thesis.summary)
            pdf.ln(4)

        # --- Data sources ---
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Data Sources", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)

        for r in response.analysis.results:
            status = "OK" if r.success else f"FAILED ({r.error})"
            pdf.cell(0, 6, f"  {r.tool} ({r.source}): {status}", new_x="LMARGIN", new_y="NEXT")

        pdf.output(str(path))
        logger.info("Saved PDF report to %s", path)
        return path
