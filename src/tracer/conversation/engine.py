"""ConversationEngine — top-level orchestrator for the conversational pipeline.

Wires together IntentParser, dispatcher, AnalysisLoop, and synthesizers
to process natural-language queries end-to-end.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from tracer.config.models import PortfolioConfig
from tracer.conversation.analysis_loop import (
    CONFIDENCE_THRESHOLD,
    MAX_ITERATIONS,
    AnalysisLoop,
    AnalysisResult,
)
from tracer.conversation.context import ConversationContext, extract_context, resolve_pronoun
from tracer.conversation.dispatcher import invoke_tools
from tracer.conversation.intent import Intent, IntentParser, IntentType
from tracer.conversation.report_exporter import ReportExporter
from tracer.conversation.synthesizer import ComparisonSynthesizer, ResponseSynthesizer
from tracer.data.registry import DataRegistry, build_registry
from tracer.llm.registry import LLMRegistry
from tracer.memory.session_compactor import SessionCompactor
from tracer.memory.session_logger import SessionLogger, TurnRecord
from tracer.models import ToolResult, TradeThesis
from tracer.tools import pipeline

logger = logging.getLogger(__name__)


@dataclass
class EngineResponse:
    """Final response from the ConversationEngine."""

    text: str
    intent: Intent
    analysis: AnalysisResult
    generated_at: datetime = field(default_factory=datetime.now)


class ConversationEngine:
    """Top-level orchestrator that wires together the conversational pipeline.

    Flow: IntentParser -> dispatcher -> AnalysisLoop -> Synthesizer
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
        report_dir: Path | None = None,
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
        self._comparison_synthesizer = ComparisonSynthesizer(llm_registry)
        self._data = data_registry
        self._portfolio_config = portfolio_config or PortfolioConfig()
        self._history: list[dict] = []
        self._session_logger = session_logger
        self._compactor = SessionCompactor(llm_registry) if session_logger else None
        self._report_exporter = ReportExporter(report_dir) if report_dir else None
        self._context: ConversationContext = ConversationContext()
        self._turn_counter = 0
        self._last_response: EngineResponse | None = None

    @property
    def history(self) -> list[dict]:
        """Turn history for the current session."""
        return list(self._history)

    def save_last_report(self, fmt: str = "md") -> Path | None:
        """Save the last analysis result as a report file.

        Args:
            fmt: ``"md"`` for Markdown, ``"json"`` for JSON.

        Returns:
            Path to the saved file, or None if no report exporter is
            configured or no previous response exists.
        """
        if self._report_exporter is None or self._last_response is None:
            return None
        resp = self._last_response
        if fmt == "json":
            return self._report_exporter.save_json(resp.intent, resp.analysis, resp.text)
        return self._report_exporter.save_markdown(resp.intent, resp.analysis, resp.text)

    def _log_turn(self, role: str, content: str, **kwargs: object) -> None:
        """Append a turn to the session log if a logger is configured."""
        if self._session_logger is None:
            return
        self._turn_counter += 1
        self._session_logger.append(
            TurnRecord(turn=self._turn_counter, role=role, content=content, **kwargs)  # type: ignore[arg-type]
        )

    async def _maybe_compact(self) -> None:
        """Trigger compaction if the session log exceeds the token threshold."""
        if self._compactor is None or self._session_logger is None:
            return
        if self._compactor.needs_compaction(self._session_logger):
            try:
                result = await self._compactor.compact(self._session_logger)
                logger.info(
                    "Session compacted: %d turns → %d tokens summary",
                    result.turn_count,
                    result.output_tokens,
                )
            except Exception:
                logger.warning("Session compaction failed", exc_info=True)

    async def query(self, user_input: str) -> EngineResponse:
        """Process a user query through the full pipeline."""
        self._history.append({"role": "user", "content": user_input})
        self._log_turn("user", user_input)

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

        # 2. Comparison branch: per-ticker analysis + comparison table.
        if intent.intent_type == IntentType.COMPARISON and len(intent.tickers) >= 2:
            response = await self._handle_comparison(intent)
        else:
            # 3. Standard analysis path.
            response = await self._handle_standard(intent)

        self._last_response = response
        return response

    async def _handle_comparison(self, intent: Intent) -> EngineResponse:
        """Run per-ticker analysis concurrently and synthesize comparison."""
        single_intents = [
            Intent(
                intent_type=IntentType.COMPARISON,
                tickers=[ticker],
                tools=intent.tools,
                raw_query=intent.raw_query,
            )
            for ticker in intent.tickers
        ]
        gathered = await asyncio.gather(
            *[invoke_tools(si.tools, si, self._data) for si in single_intents]
        )
        per_ticker_results: dict[str, list[ToolResult]] = dict(zip(intent.tickers, gathered))
        all_results = [r for results in per_ticker_results.values() for r in results]
        analysis = AnalysisResult(results=all_results, confidence=0.7, iterations=1)
        text = await self._comparison_synthesizer.synthesize(intent, per_ticker_results)
        self._history.append({"role": "assistant", "content": text})
        self._log_turn("assistant", text)
        await self._maybe_compact()
        return EngineResponse(text=text, intent=intent, analysis=analysis)

    async def _handle_standard(self, intent: Intent) -> EngineResponse:
        """Run the standard analysis pipeline (DeepPath)."""
        # Invoke initial pipeline tools.
        initial_results = await invoke_tools(intent.tools, intent, self._data)

        # Run analysis loop.
        analysis = await self._analysis_loop.run(intent, initial_results)
        logger.info(
            "Analysis complete: confidence=%.2f iterations=%d",
            analysis.confidence,
            analysis.iterations,
        )

        # Trade thesis generation (step 7) — only when tickers are present.
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

        # Risk check (step 8) — only when a trade thesis was produced.
        if analysis.trade_thesis is not None and self._portfolio_config.holdings:
            risk_result = await pipeline.risk_check(
                analysis.trade_thesis.ticker,
                analysis.trade_thesis,
                self._data,
                self._portfolio_config,
            )
            analysis.results.append(risk_result)

        # Synthesize response.
        text = await self._synthesizer.synthesize(intent, analysis)
        self._history.append({"role": "assistant", "content": text})
        self._log_turn("assistant", text)
        await self._maybe_compact()

        return EngineResponse(text=text, intent=intent, analysis=analysis)
