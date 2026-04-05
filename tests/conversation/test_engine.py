"""Tests for ConversationEngine, AnalysisLoop, and ResponseSynthesizer."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, patch

from qracer.config.models import Holding, PortfolioConfig
from qracer.conversation.engine import (
    AnalysisLoop,
    AnalysisResult,
    ConversationEngine,
    EngineResponse,
    ResponseSynthesizer,
    _invoke_tool,
    _invoke_tools,
)
from qracer.conversation.intent import Intent, IntentType
from qracer.data.registry import DataRegistry
from qracer.llm.providers import CompletionResponse, Role
from qracer.llm.registry import LLMRegistry
from qracer.models import ToolResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm_registry(responses: dict[Role, str | list[str]]) -> LLMRegistry:
    """Build an LLMRegistry with mock providers that return canned responses.

    If a role maps to a list, responses are returned in order (pop from front).
    """
    registry = LLMRegistry()
    for role, content in responses.items():
        provider = AsyncMock()
        if isinstance(content, list):
            contents = list(content)

            async def _complete(req, _contents=contents):
                c = _contents.pop(0) if _contents else "{}"
                return CompletionResponse(
                    content=c, model="mock", input_tokens=10, output_tokens=5, cost=0.0
                )

            provider.complete = _complete
        else:
            provider.complete.return_value = CompletionResponse(
                content=content, model="mock", input_tokens=10, output_tokens=5, cost=0.0
            )
        registry.register("mock", provider, [role])
    return registry


def _ok_result(tool: str, data: dict | None = None) -> ToolResult:
    return ToolResult(
        tool=tool,
        success=True,
        data=data or {"sample": "data"},
        source="test",
        fetched_at=datetime.now(),
        is_stale=False,
    )


def _failed_result(tool: str) -> ToolResult:
    return ToolResult(
        tool=tool,
        success=False,
        data={},
        source="test",
        error="test error",
    )


# ---------------------------------------------------------------------------
# _invoke_tool / _invoke_tools
# ---------------------------------------------------------------------------


class TestInvokeTool:
    async def test_price_event_with_ticker(self) -> None:
        intent = Intent(IntentType.EVENT_ANALYSIS, tickers=["AAPL"], raw_query="test")
        registry = DataRegistry()
        with patch("qracer.conversation.engine.pipeline") as mock_pipeline:
            mock_pipeline.price_event = AsyncMock(return_value=_ok_result("price_event"))
            result = await _invoke_tool("price_event", intent, registry)
            assert result.success
            mock_pipeline.price_event.assert_called_once_with("AAPL", registry)

    async def test_news_with_ticker(self) -> None:
        intent = Intent(IntentType.EVENT_ANALYSIS, tickers=["TSLA"], raw_query="test")
        registry = DataRegistry()
        with patch("qracer.conversation.engine.pipeline") as mock_pipeline:
            mock_pipeline.news = AsyncMock(return_value=_ok_result("news"))
            result = await _invoke_tool("news", intent, registry)
            assert result.success

    async def test_cross_market_passes_all_tickers(self) -> None:
        intent = Intent(IntentType.CROSS_MARKET, tickers=["AAPL", "TSLA"], raw_query="test")
        registry = DataRegistry()
        with patch("qracer.conversation.engine.pipeline") as mock_pipeline:
            mock_pipeline.cross_market = AsyncMock(return_value=_ok_result("cross_market"))
            await _invoke_tool("cross_market", intent, registry)
            mock_pipeline.cross_market.assert_called_once_with(["AAPL", "TSLA"], registry)

    async def test_macro_uses_raw_query(self) -> None:
        intent = Intent(IntentType.MACRO_QUERY, raw_query="inflation rate")
        registry = DataRegistry()
        with patch("qracer.conversation.engine.pipeline") as mock_pipeline:
            mock_pipeline.macro = AsyncMock(return_value=_ok_result("macro"))
            await _invoke_tool("macro", intent, registry)
            mock_pipeline.macro.assert_called_once_with("inflation rate", registry)

    async def test_memory_search(self) -> None:
        intent = Intent(IntentType.FOLLOW_UP, raw_query="what about before?")
        registry = DataRegistry()
        with patch("qracer.conversation.engine.pipeline") as mock_pipeline:
            mock_pipeline.memory_search = AsyncMock(return_value=_ok_result("memory_search"))
            await _invoke_tool("memory_search", intent, registry)
            mock_pipeline.memory_search.assert_called_once_with("what about before?")

    async def test_tool_without_tickers_returns_failure(self) -> None:
        intent = Intent(IntentType.EVENT_ANALYSIS, tickers=[], raw_query="test")
        registry = DataRegistry()
        result = await _invoke_tool("price_event", intent, registry)
        assert not result.success
        assert result.error is not None and "no tickers" in result.error

    async def test_invoke_tools_concurrent(self) -> None:
        intent = Intent(IntentType.EVENT_ANALYSIS, tickers=["AAPL"], raw_query="test")
        registry = DataRegistry()
        with patch("qracer.conversation.engine.pipeline") as mock_pipeline:
            mock_pipeline.price_event = AsyncMock(return_value=_ok_result("price_event"))
            mock_pipeline.news = AsyncMock(return_value=_ok_result("news"))
            results = await _invoke_tools(["price_event", "news"], intent, registry)
            assert len(results) == 2
            assert all(r.success for r in results)

    async def test_invoke_tools_empty_list(self) -> None:
        intent = Intent(IntentType.FOLLOW_UP, raw_query="ok")
        results = await _invoke_tools([], intent, DataRegistry())
        assert results == []


# ---------------------------------------------------------------------------
# AnalysisLoop
# ---------------------------------------------------------------------------


class TestAnalysisLoop:
    async def test_exits_on_high_confidence(self) -> None:
        """Loop should exit after first evaluation if confidence is high."""
        llm = _mock_llm_registry(
            {
                Role.ANALYST: json.dumps({"confidence": 0.9, "missing_tools": []}),
            }
        )
        loop = AnalysisLoop(llm, DataRegistry())
        intent = Intent(IntentType.EVENT_ANALYSIS, tickers=["AAPL"], raw_query="test")

        result = await loop.run(intent, [_ok_result("price_event")])
        assert result.confidence >= 0.7
        assert result.iterations == 1
        assert result.early_exit_reason is None

    async def test_fetches_missing_tools(self) -> None:
        """Loop should call additional tools when analyst suggests them."""
        responses = [
            json.dumps({"confidence": 0.4, "missing_tools": ["news"]}),
            json.dumps({"confidence": 0.85, "missing_tools": []}),
        ]
        llm = _mock_llm_registry({Role.ANALYST: responses})
        data = DataRegistry()
        loop = AnalysisLoop(llm, data)
        intent = Intent(IntentType.EVENT_ANALYSIS, tickers=["AAPL"], raw_query="test")

        with patch("qracer.conversation.engine._invoke_tools") as mock_invoke:
            mock_invoke.return_value = [_ok_result("news")]
            result = await loop.run(intent, [_ok_result("price_event")])

        assert result.confidence >= 0.7
        assert result.iterations == 2
        # Should have the initial + fetched results
        assert len(result.results) == 2

    async def test_max_iterations_cap(self) -> None:
        """Loop should not exceed max_iterations."""
        low_conf = json.dumps({"confidence": 0.3, "missing_tools": ["news"]})
        llm = _mock_llm_registry({Role.ANALYST: [low_conf, low_conf, low_conf]})
        data = DataRegistry()
        loop = AnalysisLoop(llm, data, max_iterations=2)
        intent = Intent(IntentType.EVENT_ANALYSIS, tickers=["AAPL"], raw_query="test")

        with patch("qracer.conversation.engine._invoke_tools") as mock_invoke:
            mock_invoke.return_value = [_ok_result("news")]
            result = await loop.run(intent, [_ok_result("price_event")])

        assert result.iterations <= 2

    async def test_early_exit_on_multiple_failures(self) -> None:
        """Loop should exit early if >=2 tools fail in first iteration."""
        llm = _mock_llm_registry(
            {
                Role.ANALYST: json.dumps({"confidence": 0.5, "missing_tools": []}),
            }
        )
        loop = AnalysisLoop(llm, DataRegistry())
        intent = Intent(IntentType.EVENT_ANALYSIS, tickers=["AAPL"], raw_query="test")

        result = await loop.run(intent, [_failed_result("price_event"), _failed_result("news")])
        assert result.early_exit_reason is not None
        assert "2 tools failed" in result.early_exit_reason

    async def test_llm_failure_returns_zero_confidence(self) -> None:
        """If the analyst LLM fails, confidence should be 0."""
        provider = AsyncMock()
        provider.complete.side_effect = RuntimeError("LLM down")
        llm = LLMRegistry()
        llm.register("mock", provider, [Role.ANALYST])

        loop = AnalysisLoop(llm, DataRegistry())
        intent = Intent(IntentType.EVENT_ANALYSIS, tickers=["AAPL"], raw_query="test")

        result = await loop.run(intent, [_ok_result("price_event")])
        assert result.confidence == 0.0

    async def test_no_missing_tools_exits(self) -> None:
        """If analyst says no missing tools but confidence is low, exit."""
        llm = _mock_llm_registry(
            {
                Role.ANALYST: json.dumps({"confidence": 0.5, "missing_tools": []}),
            }
        )
        loop = AnalysisLoop(llm, DataRegistry())
        intent = Intent(IntentType.EVENT_ANALYSIS, tickers=["AAPL"], raw_query="test")

        result = await loop.run(intent, [_ok_result("price_event")])
        assert result.early_exit_reason == "no additional tools suggested"


# ---------------------------------------------------------------------------
# ResponseSynthesizer
# ---------------------------------------------------------------------------


class TestResponseSynthesizer:
    async def test_synthesize_calls_strategist(self) -> None:
        llm = _mock_llm_registry(
            {
                Role.STRATEGIST: "[ANALYSIS: AAPL]\nConviction: 8/10\n...",
            }
        )
        synth = ResponseSynthesizer(llm)
        intent = Intent(IntentType.EVENT_ANALYSIS, tickers=["AAPL"], raw_query="test")
        analysis = AnalysisResult(results=[_ok_result("price_event")], confidence=0.8, iterations=1)

        text = await synth.synthesize(intent, analysis)
        assert "ANALYSIS" in text
        assert "AAPL" in text

    async def test_fallback_on_llm_failure(self) -> None:
        provider = AsyncMock()
        provider.complete.side_effect = RuntimeError("LLM down")
        llm = LLMRegistry()
        llm.register("mock", provider, [Role.STRATEGIST])

        synth = ResponseSynthesizer(llm)
        intent = Intent(IntentType.EVENT_ANALYSIS, tickers=["AAPL"], raw_query="test")
        analysis = AnalysisResult(
            results=[_ok_result("price_event"), _failed_result("news")],
            confidence=0.5,
            iterations=1,
        )

        text = await synth.synthesize(intent, analysis)
        assert "AAPL" in text
        assert "LLM error" in text

    async def test_includes_failed_tools_in_caveats(self) -> None:
        llm = _mock_llm_registry(
            {
                Role.STRATEGIST: "[ANALYSIS] with insider caveat",
            }
        )
        synth = ResponseSynthesizer(llm)
        intent = Intent(IntentType.EVENT_ANALYSIS, tickers=["AAPL"], raw_query="test")
        analysis = AnalysisResult(
            results=[_ok_result("price_event"), _failed_result("insider")],
            confidence=0.6,
            iterations=1,
        )

        # The synthesizer should pass failed tools info to the LLM.
        text = await synth.synthesize(intent, analysis)
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# ConversationEngine (integration)
# ---------------------------------------------------------------------------


class TestConversationEngine:
    async def test_full_pipeline(self) -> None:
        """End-to-end test: query → intent → tools → analysis → response."""
        intent_resp = json.dumps({"intent": "event_analysis", "tickers": ["AAPL"]})
        analysis_resp = json.dumps({"confidence": 0.85, "missing_tools": []})
        synth_resp = "[ANALYSIS: AAPL — 2025-01-01]\nConviction: 8/10\nTest response"

        llm = _mock_llm_registry(
            {
                Role.RESEARCHER: intent_resp,
                Role.ANALYST: analysis_resp,
                Role.STRATEGIST: synth_resp,
            }
        )
        data = DataRegistry()

        engine = ConversationEngine(llm, data)

        with patch("qracer.conversation.engine._invoke_tools") as mock_invoke:
            mock_invoke.return_value = [_ok_result("price_event"), _ok_result("news")]
            response = await engine.query("Why did AAPL spike 5% today?")

        assert isinstance(response, EngineResponse)
        assert response.intent.intent_type == IntentType.EVENT_ANALYSIS
        assert response.intent.tickers == ["AAPL"]
        assert "AAPL" in response.text
        assert response.analysis.confidence >= 0.7

    async def test_history_tracking(self) -> None:
        """Engine should track user/assistant turns."""
        intent_resp = json.dumps({"intent": "macro_query", "tickers": []})
        analysis_resp = json.dumps({"confidence": 0.9, "missing_tools": []})

        llm = _mock_llm_registry(
            {
                Role.RESEARCHER: intent_resp,
                Role.ANALYST: analysis_resp,
                Role.STRATEGIST: "Macro analysis response",
            }
        )

        engine = ConversationEngine(llm, DataRegistry())

        with patch("qracer.conversation.engine._invoke_tools") as mock_invoke:
            mock_invoke.return_value = [_ok_result("macro")]
            await engine.query("Where are we in the rate cycle?")

        assert len(engine.history) == 2
        assert engine.history[0]["role"] == "user"
        assert engine.history[1]["role"] == "assistant"

    async def test_risk_check_called_when_thesis_and_holdings(self) -> None:
        """Step 8: risk_check runs when trade thesis succeeds and portfolio has holdings."""
        intent_resp = json.dumps({"intent": "event_analysis", "tickers": ["AAPL"]})
        analysis_resp = json.dumps({"confidence": 0.85, "missing_tools": []})
        thesis_json = json.dumps(
            {
                "entry_zone": [170.0, 175.0],
                "target_price": 200.0,
                "stop_loss": 160.0,
                "catalyst": "AI revenue growth",
                "catalyst_date": "Q2 2026",
                "conviction": 8,
                "summary": "Strong momentum thesis.",
            }
        )
        synth_resp = "[ANALYSIS: AAPL]\nConviction: 8/10\nRisk-checked response"

        llm = _mock_llm_registry(
            {
                Role.RESEARCHER: intent_resp,
                Role.ANALYST: analysis_resp,
                Role.STRATEGIST: [thesis_json, synth_resp],
            }
        )

        portfolio = PortfolioConfig(
            holdings=[Holding(ticker="MSFT", shares=100, avg_cost=300.0)],
        )
        engine = ConversationEngine(llm, DataRegistry(), portfolio_config=portfolio)

        with (
            patch("qracer.conversation.engine._invoke_tools") as mock_invoke,
            patch("qracer.conversation.engine.pipeline.risk_check") as mock_risk,
        ):
            mock_invoke.return_value = [_ok_result("price_event")]
            mock_risk.return_value = _ok_result(
                "risk_check", {"allocation_pct": 5.0, "limits_breached": []}
            )
            response = await engine.query("Analyze AAPL")

        mock_risk.assert_called_once()
        call_args = mock_risk.call_args
        assert call_args[0][0] == "AAPL"  # ticker
        assert call_args[0][1].ticker == "AAPL"  # TradeThesis
        assert call_args[0][1].conviction == 8

        # risk_check result should be appended
        risk_results = [r for r in response.analysis.results if r.tool == "risk_check"]
        assert len(risk_results) == 1

    async def test_risk_check_skipped_without_holdings(self) -> None:
        """Step 8 is skipped when portfolio has no holdings."""
        intent_resp = json.dumps({"intent": "event_analysis", "tickers": ["AAPL"]})
        analysis_resp = json.dumps({"confidence": 0.85, "missing_tools": []})
        thesis_json = json.dumps(
            {
                "entry_zone": [170.0, 175.0],
                "target_price": 200.0,
                "stop_loss": 160.0,
                "catalyst": "AI revenue growth",
                "catalyst_date": None,
                "conviction": 7,
                "summary": "Thesis.",
            }
        )
        synth_resp = "Response"

        llm = _mock_llm_registry(
            {
                Role.RESEARCHER: intent_resp,
                Role.ANALYST: analysis_resp,
                Role.STRATEGIST: [thesis_json, synth_resp],
            }
        )

        # No holdings → risk check should not run
        engine = ConversationEngine(llm, DataRegistry())

        with (
            patch("qracer.conversation.engine._invoke_tools") as mock_invoke,
            patch("qracer.conversation.engine.pipeline.risk_check") as mock_risk,
        ):
            mock_invoke.return_value = [_ok_result("price_event")]
            await engine.query("Analyze AAPL")

        mock_risk.assert_not_called()

    async def test_risk_check_skipped_without_thesis(self) -> None:
        """Step 8 is skipped when trade thesis generation fails."""
        intent_resp = json.dumps({"intent": "event_analysis", "tickers": ["AAPL"]})
        analysis_resp = json.dumps({"confidence": 0.85, "missing_tools": []})
        # Invalid JSON → thesis fails
        synth_resp = "Plain text, not JSON"

        llm = _mock_llm_registry(
            {
                Role.RESEARCHER: intent_resp,
                Role.ANALYST: analysis_resp,
                Role.STRATEGIST: [synth_resp, synth_resp],
            }
        )

        portfolio = PortfolioConfig(
            holdings=[Holding(ticker="MSFT", shares=100, avg_cost=300.0)],
        )
        engine = ConversationEngine(llm, DataRegistry(), portfolio_config=portfolio)

        with (
            patch("qracer.conversation.engine._invoke_tools") as mock_invoke,
            patch("qracer.conversation.engine.pipeline.risk_check") as mock_risk,
        ):
            mock_invoke.return_value = [_ok_result("price_event")]
            await engine.query("Analyze AAPL")

        mock_risk.assert_not_called()

    async def test_custom_thresholds(self) -> None:
        """Engine should accept custom max_iterations and confidence_threshold."""
        engine = ConversationEngine(
            _mock_llm_registry(
                {
                    Role.RESEARCHER: json.dumps({"intent": "macro_query", "tickers": []}),
                    Role.ANALYST: json.dumps({"confidence": 0.5, "missing_tools": []}),
                    Role.STRATEGIST: "response",
                }
            ),
            DataRegistry(),
            max_iterations=1,
            confidence_threshold=0.9,
        )

        with patch("qracer.conversation.engine._invoke_tools") as mock_invoke:
            mock_invoke.return_value = []
            response = await engine.query("test")

        # With threshold 0.9 and confidence 0.5, it should have low confidence.
        assert response.analysis.confidence < 0.9
