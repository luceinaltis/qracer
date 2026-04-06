"""AnalysisLoop — iteratively gathers data until confidence threshold."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from qracer.conversation.dispatcher import TOOL_DISPATCH, invoke_tools
from qracer.conversation.intent import Intent
from qracer.data.registry import DataRegistry
from qracer.llm.providers import CompletionRequest, CompletionResponse, Message, Role
from qracer.llm.registry import LLMRegistry
from qracer.models import ToolResult, TradeThesis

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3
CONFIDENCE_THRESHOLD = 0.7


@dataclass
class AnalysisResult:
    """Output of the AnalysisLoop."""

    results: list[ToolResult] = field(default_factory=list)
    confidence: float = 0.0
    iterations: int = 0
    early_exit_reason: str | None = None
    trade_thesis: TradeThesis | None = None


class AnalysisLoop:
    """Iteratively gathers data until confidence threshold or max iterations.

    Each iteration:
    1. Evaluate confidence from collected evidence.
    2. If confidence >= threshold or iterations exhausted -> exit.
    3. Otherwise, ask the analyst LLM what data is missing and fetch it.
    """

    def __init__(
        self,
        llm_registry: LLMRegistry,
        data_registry: DataRegistry,
        *,
        max_iterations: int = MAX_ITERATIONS,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self._llm = llm_registry
        self._data = data_registry
        self._max_iterations = max_iterations
        self._confidence_threshold = confidence_threshold

    async def run(self, intent: Intent, initial_results: list[ToolResult]) -> AnalysisResult:
        """Run the analysis loop starting from initial tool results."""
        all_results = list(initial_results)
        iterations = 0

        for iteration in range(self._max_iterations):
            iterations = iteration + 1

            successful = [r for r in all_results if r.success]
            failed = [r for r in all_results if not r.success]

            if len(failed) >= 2 and iteration == 0:
                return AnalysisResult(
                    results=all_results,
                    confidence=0.0,
                    iterations=iterations,
                    early_exit_reason=">=2 tools failed in first iteration",
                )

            confidence, missing_tools = await self._evaluate(intent, successful)

            if confidence >= self._confidence_threshold:
                return AnalysisResult(
                    results=all_results,
                    confidence=confidence,
                    iterations=iterations,
                )

            if not missing_tools or iteration == self._max_iterations - 1:
                return AnalysisResult(
                    results=all_results,
                    confidence=confidence,
                    iterations=iterations,
                    early_exit_reason="no additional tools suggested"
                    if not missing_tools
                    else None,
                )

            new_results = await invoke_tools(missing_tools, intent, self._data)
            all_results.extend(new_results)

        return AnalysisResult(
            results=all_results,
            confidence=0.0,
            iterations=iterations,
        )

    async def _evaluate(self, intent: Intent, results: list[ToolResult]) -> tuple[float, list[str]]:
        """Ask the analyst LLM to evaluate confidence and suggest missing tools."""
        evidence = format_evidence(results)
        system = (
            "You are an analyst evaluating whether collected data is sufficient to "
            "answer a financial query.\n"
            "Available tools: price_event, news, insider, macro, fundamentals, "
            "cross_market, memory_search.\n"
            'Return JSON: {"confidence": 0.0-1.0, "missing_tools": [...]}\n'
            "confidence = how confident you are the data answers the query.\n"
            "missing_tools = tools not yet called that would improve the answer "
            "(empty list if sufficient).\n"
            "Return ONLY valid JSON."
        )
        user_msg = f"Query: {intent.raw_query}\n\nEvidence collected:\n{evidence}"

        try:
            provider = self._llm.get(Role.ANALYST)
            response: CompletionResponse = await provider.complete(
                CompletionRequest(
                    messages=[
                        Message(role="system", content=system),
                        Message(role="user", content=user_msg),
                    ],
                    max_tokens=256,
                    temperature=0.0,
                )
            )
            parsed = json.loads(response.content)
            confidence = float(parsed.get("confidence", 0.0))
            missing = [t for t in parsed.get("missing_tools", []) if t in TOOL_DISPATCH]
            already_called = {r.tool for r in results}
            missing = [t for t in missing if t not in already_called]
            return confidence, missing
        except Exception:
            logger.warning("AnalysisLoop evaluation failed", exc_info=True)
            return 0.0, []


def format_evidence(results: list[ToolResult]) -> str:
    """Format tool results into a prompt-friendly string."""
    if not results:
        return "(no data collected)"
    sections: list[str] = []
    for r in results:
        sections.append(
            f"[{r.tool}] source={r.source} stale={r.is_stale}\n{json.dumps(r.data, indent=2)}"
        )
    return "\n\n".join(sections)
