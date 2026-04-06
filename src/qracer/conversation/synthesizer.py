"""Response synthesizers — format analysis results for the user."""

from __future__ import annotations

import logging
from datetime import datetime

from qracer.conversation.analysis_loop import AnalysisResult, format_evidence
from qracer.conversation.intent import Intent
from qracer.llm.providers import CompletionRequest, Message, Role
from qracer.llm.registry import LLMRegistry
from qracer.models import ToolResult

logger = logging.getLogger(__name__)


class ResponseSynthesizer:
    """Synthesizes a final response from analysis results.

    Produces the spec response format: causal chain, adversarial check,
    conviction score.
    """

    def __init__(self, llm_registry: LLMRegistry) -> None:
        self._llm = llm_registry

    async def synthesize(self, intent: Intent, analysis: AnalysisResult) -> str:
        """Generate the final user-facing response."""
        evidence = format_evidence([r for r in analysis.results if r.success])
        failed_tools = [r.tool for r in analysis.results if not r.success]

        ticker_str = ", ".join(intent.tickers) if intent.tickers else "general market"
        today = datetime.now().strftime("%Y-%m-%d")

        system = (
            "You are a senior financial analyst. Synthesize the evidence into a "
            "structured response.\n\n"
            "Use this EXACT format:\n"
            f"[ANALYSIS: {ticker_str} — {today}]\n"
            "Conviction: {{score}}/10\n\n"
            "WHAT HAPPENED\n"
            "{{1-2 sentence direct answer}}\n\n"
            "EVIDENCE CHAIN\n"
            "{{numbered evidence items with source and conclusion}}\n\n"
            "ADVERSARIAL CHECK\n"
            "{{bullet points: reasons this could be wrong, data caveats}}\n\n"
            "VERDICT\n"
            "{{final judgment with conviction score and key qualifier}}"
        )

        caveats = ""
        if failed_tools:
            caveats = (
                f"\n\nNote: the following data sources were unavailable: "
                f"{', '.join(failed_tools)}. Flag this in ADVERSARIAL CHECK."
            )

        if analysis.early_exit_reason:
            caveats += (
                f"\nAnalysis exited early: {analysis.early_exit_reason}. "
                "Acknowledge data limitations."
            )

        user_msg = (
            f"Query: {intent.raw_query}\n\n"
            f"Evidence (confidence={analysis.confidence:.2f}, "
            f"iterations={analysis.iterations}):\n{evidence}{caveats}"
        )

        try:
            provider = self._llm.get(Role.STRATEGIST)
            response = await provider.complete(
                CompletionRequest(
                    messages=[
                        Message(role="system", content=system),
                        Message(role="user", content=user_msg),
                    ],
                    max_tokens=2048,
                    temperature=0.2,
                )
            )
            return response.content
        except Exception:
            logger.warning("ResponseSynthesizer failed, returning raw evidence", exc_info=True)
            return self._fallback_response(intent, analysis)

    def _fallback_response(self, intent: Intent, analysis: AnalysisResult) -> str:
        """Minimal plain-text fallback when the LLM is unavailable."""
        lines = [f"[ANALYSIS: {', '.join(intent.tickers) or 'general'}]"]
        lines.append(f"Query: {intent.raw_query}")
        lines.append(f"Confidence: {analysis.confidence:.2f}")
        lines.append("")
        for r in analysis.results:
            status = "OK" if r.success else f"FAILED ({r.error})"
            lines.append(f"  [{r.tool}] {status}")
        lines.append("")
        lines.append("(Full synthesis unavailable — LLM error)")
        return "\n".join(lines)


class ComparisonSynthesizer:
    """Synthesizes a side-by-side comparison for multiple tickers."""

    def __init__(self, llm_registry: LLMRegistry) -> None:
        self._llm = llm_registry

    async def synthesize(
        self, intent: Intent, per_ticker_results: dict[str, list[ToolResult]]
    ) -> str:
        """Generate a comparative response table with verdict."""
        evidence_sections: list[str] = []
        for ticker, results in per_ticker_results.items():
            successful = [r for r in results if r.success]
            evidence_sections.append(f"--- {ticker} ---\n{format_evidence(successful)}")
        combined_evidence = "\n\n".join(evidence_sections)
        tickers_str = ", ".join(intent.tickers)
        today = datetime.now().strftime("%Y-%m-%d")

        header = "| Metric | " + " | ".join(intent.tickers) + " |"
        separator = "|" + "---|" * (len(intent.tickers) + 1)

        system = (
            "You are a senior financial analyst. Compare the given tickers "
            "side-by-side.\n\n"
            "Use this EXACT format:\n"
            f"[COMPARISON: {tickers_str} — {today}]\n\n"
            f"{header}\n{separator}\n"
            "| Price | ... |\n"
            "| PE Ratio | ... |\n"
            "| Revenue Growth | ... |\n"
            "| Conviction | ... |\n"
            "(add more rows as the data supports)\n\n"
            "VERDICT\n"
            "{{comparative analysis: which ticker is stronger and why, "
            "1-3 sentences}}"
        )
        user_msg = f"Query: {intent.raw_query}\n\nEvidence per ticker:\n{combined_evidence}"

        try:
            provider = self._llm.get(Role.STRATEGIST)
            response = await provider.complete(
                CompletionRequest(
                    messages=[
                        Message(role="system", content=system),
                        Message(role="user", content=user_msg),
                    ],
                    max_tokens=2048,
                    temperature=0.2,
                )
            )
            return response.content
        except Exception:
            logger.warning("ComparisonSynthesizer failed, returning fallback", exc_info=True)
            return self._fallback_response(intent, per_ticker_results)

    def _fallback_response(
        self, intent: Intent, per_ticker_results: dict[str, list[ToolResult]]
    ) -> str:
        lines = [f"[COMPARISON: {', '.join(intent.tickers)}]"]
        lines.append(f"Query: {intent.raw_query}")
        lines.append("")
        for ticker, results in per_ticker_results.items():
            lines.append(f"  {ticker}:")
            for r in results:
                status = "OK" if r.success else f"FAILED ({r.error})"
                lines.append(f"    [{r.tool}] {status}")
        lines.append("")
        lines.append("(Full comparison unavailable — LLM error)")
        return "\n".join(lines)
