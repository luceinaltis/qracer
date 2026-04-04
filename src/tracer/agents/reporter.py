"""Reporter agent — alpha report generation.

Pipeline step: Alpha Report (step 7).
"""

from __future__ import annotations

from tracer.agents.base import BaseAgent
from tracer.llm.providers import Role
from tracer.models import Report, Signal


class Reporter(BaseAgent):
    """Summary and report generation.

    Responsibilities:
    - Generate actionable alpha reports from scored signals.
    - Format as "What the market doesn't know yet" narrative.
    """

    role = Role.REPORTER

    async def generate_report(
        self,
        signals: list[Signal],
        *,
        context: dict | None = None,
    ) -> Report:
        """Generate an alpha report from scored signals.

        Args:
            signals: Ranked signals from the Strategist.
            context: Optional extra context (regime, cross-market data).

        Returns:
            A Report with thesis, evidence chain, adversarial check, and verdict.
        """
        import json

        if not signals:
            return Report(
                title="No actionable signals",
                ticker=None,
                conviction=0.0,
                what_happened="Insufficient data or no contrarian signals detected.",
                evidence_chain=[],
                adversarial_check=["No signals to evaluate."],
                verdict="No trade recommended.",
            )

        # Use the highest-conviction signal as the primary focus
        primary = signals[0]

        signals_data = [
            {
                "ticker": s.ticker,
                "direction": s.direction.value,
                "conviction": s.conviction,
                "thesis": s.thesis,
                "evidence": s.evidence,
                "contrarian_angle": s.contrarian_angle,
                "risk_factors": s.risk_factors,
                "time_horizon_days": s.time_horizon_days,
            }
            for s in signals
        ]

        context_text = ""
        if context:
            context_text = f"\n\nAdditional context:\n{json.dumps(context, indent=2)}"

        response = await self._complete(
            system=(
                "You are a financial report writer. Generate an alpha report in "
                "this exact JSON format:\n"
                "{\n"
                '  "title": "concise title",\n'
                '  "what_happened": "1-2 sentence direct answer",\n'
                '  "evidence_chain": ["evidence1 — Source: X", ...],\n'
                '  "adversarial_check": ["reason this could be wrong", ...],\n'
                '  "verdict": "final judgment with conviction qualifier"\n'
                "}\n\n"
                "Frame the narrative as 'What the market doesn't know yet'. "
                "Be specific, cite evidence, and include risk factors."
            ),
            user=(
                f"Generate an alpha report for these signals:\n"
                f"{json.dumps(signals_data, indent=2)}{context_text}"
            ),
        )

        try:
            report_data = json.loads(response.content)
        except (json.JSONDecodeError, ValueError):
            report_data = {}

        return Report(
            title=report_data.get("title", f"Alpha Report: {primary.ticker}"),
            ticker=primary.ticker,
            conviction=primary.conviction,
            what_happened=report_data.get("what_happened", primary.thesis),
            evidence_chain=report_data.get("evidence_chain", primary.evidence),
            adversarial_check=report_data.get(
                "adversarial_check", primary.risk_factors
            ),
            verdict=report_data.get("verdict", primary.thesis),
            signals=signals,
        )

    async def run(self, signals: list[Signal], **kwargs) -> Report:
        """Default entry point — generate report from signals."""
        return await self.generate_report(signals, **kwargs)
