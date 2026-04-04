"""ConversationEngine — orchestrates the full conversational pipeline.

Wires together IntentParser, pipeline tools, AnalysisLoop, and
ResponseSynthesizer to process natural-language queries end-to-end.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime

from tracer.config.models import PortfolioConfig
from tracer.conversation.context import ConversationContext, extract_context, resolve_pronoun
from tracer.conversation.intent import Intent, IntentParser
from tracer.data.registry import DataRegistry, build_registry
from tracer.llm.providers import CompletionRequest, CompletionResponse, Message, Role
from tracer.llm.registry import LLMRegistry
from tracer.memory.session_logger import SessionLogger
from tracer.models import ToolResult, TradeThesis
from tracer.tools import pipeline

logger = logging.getLogger(__name__)

# AnalysisLoop defaults
MAX_ITERATIONS = 3
CONFIDENCE_THRESHOLD = 0.7

# Dispatcher: maps tool name → callable signature
_TOOL_DISPATCH: dict[str, str] = {
    "price_event": "ticker",
    "news": "ticker",
    "insider": "ticker",
    "macro": "indicator",
    "fundamentals": "ticker",
    "cross_market": "tickers",
    "memory_search": "query",
}


@dataclass
class AnalysisResult:
    """Output of the AnalysisLoop."""

    results: list[ToolResult] = field(default_factory=list)
    confidence: float = 0.0
    iterations: int = 0
    early_exit_reason: str | None = None
    trade_thesis: TradeThesis | None = None


@dataclass
class EngineResponse:
    """Final response from the ConversationEngine."""

    text: str
    intent: Intent
    analysis: AnalysisResult
    generated_at: datetime = field(default_factory=datetime.now)


async def _invoke_tool(tool_name: str, intent: Intent, registry: DataRegistry) -> ToolResult:
    """Invoke a single pipeline tool based on the intent context."""
    tickers = intent.tickers

    if tool_name == "price_event" and tickers:
        return await pipeline.price_event(tickers[0], registry)
    if tool_name == "news" and tickers:
        return await pipeline.news(tickers[0], registry)
    if tool_name == "insider" and tickers:
        return await pipeline.insider(tickers[0], registry)
    if tool_name == "fundamentals" and tickers:
        return await pipeline.fundamentals(tickers[0], registry)
    if tool_name == "cross_market" and tickers:
        return await pipeline.cross_market(tickers, registry)
    if tool_name == "macro":
        # Use the raw query as the indicator hint for macro lookups.
        return await pipeline.macro(intent.raw_query, registry)
    if tool_name == "memory_search":
        return await pipeline.memory_search(intent.raw_query)

    # Tool requires tickers but none available — return a failed result.
    return ToolResult(
        tool=tool_name,
        success=False,
        data={},
        source="dispatcher",
        error=f"Cannot invoke {tool_name}: no tickers in query",
    )


async def _invoke_tools(
    tool_names: list[str], intent: Intent, registry: DataRegistry
) -> list[ToolResult]:
    """Invoke multiple pipeline tools concurrently."""
    if not tool_names:
        return []
    coros = [_invoke_tool(name, intent, registry) for name in tool_names]
    return list(await asyncio.gather(*coros))


class AnalysisLoop:
    """Iteratively gathers data until confidence threshold or max iterations.

    Each iteration:
    1. Evaluate confidence from collected evidence.
    2. If confidence >= threshold or iterations exhausted → exit.
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

            # Check how many tools failed in this batch.
            successful = [r for r in all_results if r.success]
            failed = [r for r in all_results if not r.success]

            if len(failed) >= 2 and iteration == 0:
                return AnalysisResult(
                    results=all_results,
                    confidence=0.0,
                    iterations=iterations,
                    early_exit_reason=">=2 tools failed in first iteration",
                )

            # Ask analyst to evaluate confidence and decide next steps.
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

            # Fetch missing data.
            new_results = await _invoke_tools(missing_tools, intent, self._data)
            all_results.extend(new_results)

        return AnalysisResult(
            results=all_results,
            confidence=0.0,
            iterations=iterations,
        )

    async def _evaluate(self, intent: Intent, results: list[ToolResult]) -> tuple[float, list[str]]:
        """Ask the analyst LLM to evaluate confidence and suggest missing tools.

        Returns (confidence, list_of_missing_tool_names).
        """
        evidence = _format_evidence(results)
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
            missing = [t for t in parsed.get("missing_tools", []) if t in _TOOL_DISPATCH]
            # Don't re-call tools we already have results for.
            already_called = {r.tool for r in results}
            missing = [t for t in missing if t not in already_called]
            return confidence, missing
        except Exception:
            logger.warning("AnalysisLoop evaluation failed", exc_info=True)
            return 0.0, []


class ResponseSynthesizer:
    """Synthesizes a final response from analysis results.

    Produces the spec response format: causal chain, adversarial check,
    conviction score.
    """

    def __init__(self, llm_registry: LLMRegistry) -> None:
        self._llm = llm_registry

    async def synthesize(self, intent: Intent, analysis: AnalysisResult) -> str:
        """Generate the final user-facing response."""
        evidence = _format_evidence([r for r in analysis.results if r.success])
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


class ConversationEngine:
    """Top-level orchestrator that wires together the conversational pipeline.

    Flow: IntentParser → pipeline tools → AnalysisLoop → ResponseSynthesizer
    """

    def __init__(
        self,
        llm_registry: LLMRegistry,
        data_registry: DataRegistry | None = None,
        portfolio_config: PortfolioConfig | None = None,
        *,
        max_iterations: int = MAX_ITERATIONS,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        session_logger: SessionLogger | None = None,
    ) -> None:
        if data_registry is None:
            data_registry = build_registry()
        self._llm = llm_registry
        self._intent_parser = IntentParser(llm_registry)
        self._analysis_loop = AnalysisLoop(
            llm_registry,
            data_registry,
            max_iterations=max_iterations,
            confidence_threshold=confidence_threshold,
        )
        self._synthesizer = ResponseSynthesizer(llm_registry)
        self._data = data_registry
        self._portfolio_config = portfolio_config or PortfolioConfig()
        self._history: list[dict] = []
        self._session_logger = session_logger
        self._context: ConversationContext = ConversationContext()

    @property
    def history(self) -> list[dict]:
        """Turn history for the current session."""
        return list(self._history)

    async def query(self, user_input: str) -> EngineResponse:
        """Process a user query through the full pipeline."""
        self._history.append({"role": "user", "content": user_input})

        # 0. Extract conversation context from session log.
        if self._session_logger is not None:
            turns = self._session_logger.read_all()[-50:]
            self._context = extract_context(turns)

        # 1. Parse intent.
        intent = await self._intent_parser.parse(user_input)

        # 1b. If no tickers in intent, try resolving from conversation context.
        if not intent.tickers and self._context.current_topic:
            resolved = resolve_pronoun(user_input.strip(), self._context)
            if resolved is None:
                resolved = self._context.current_topic
            intent = Intent(
                intent_type=intent.intent_type,
                tickers=[resolved],
                tools=intent.tools,
                raw_query=intent.raw_query,
            )
        logger.info(
            "Parsed intent: %s tickers=%s tools=%s",
            intent.intent_type.value,
            intent.tickers,
            intent.tools,
        )

        # 2. Invoke initial pipeline tools.
        initial_results = await _invoke_tools(intent.tools, intent, self._data)

        # 3. Run analysis loop.
        analysis = await self._analysis_loop.run(intent, initial_results)
        logger.info(
            "Analysis complete: confidence=%.2f iterations=%d",
            analysis.confidence,
            analysis.iterations,
        )

        # 4. Trade thesis generation (step 7) — only when tickers are present.
        if intent.tickers:
            thesis_result = await pipeline.trade_thesis(
                intent.tickers[0], analysis.results, self._llm
            )
            analysis.results.append(thesis_result)
            if thesis_result.success and thesis_result.data.get("thesis"):
                td = thesis_result.data["thesis"]
                try:
                    analysis.trade_thesis = TradeThesis(
                        ticker=td["ticker"],
                        entry_zone=tuple(td["entry_zone"]),  # type: ignore[arg-type]
                        target_price=td["target_price"],
                        stop_loss=td["stop_loss"],
                        risk_reward_ratio=td["risk_reward_ratio"],
                        catalyst=td["catalyst"],
                        catalyst_date=td.get("catalyst_date"),
                        conviction=td["conviction"],
                        summary=td["summary"],
                    )
                except (KeyError, ValueError, TypeError):
                    logger.warning("Failed to reconstruct TradeThesis from result")

        # 5. Risk check (step 8) — only when a trade thesis was produced.
        if analysis.trade_thesis is not None and self._portfolio_config.holdings:
            risk_result = await pipeline.risk_check(
                analysis.trade_thesis.ticker,
                analysis.trade_thesis,
                self._data,
                self._portfolio_config,
            )
            analysis.results.append(risk_result)

        # 6. Synthesize response.
        text = await self._synthesizer.synthesize(intent, analysis)

        self._history.append({"role": "assistant", "content": text})

        return EngineResponse(
            text=text,
            intent=intent,
            analysis=analysis,
        )


def _format_evidence(results: list[ToolResult]) -> str:
    """Format tool results into a prompt-friendly string."""
    if not results:
        return "(no data collected)"
    sections: list[str] = []
    for r in results:
        sections.append(
            f"[{r.tool}] source={r.source} stale={r.is_stale}\n{json.dumps(r.data, indent=2)}"
        )
    return "\n\n".join(sections)
