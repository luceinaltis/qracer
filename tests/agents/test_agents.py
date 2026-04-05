"""Tests for Tracer agent roles."""

from __future__ import annotations

import json
from datetime import date, datetime

import pytest

from qracer.agents import Analyst, BaseAgent, Reporter, Researcher, Strategist
from qracer.data.providers import (
    OHLCV,
    FundamentalData,
    MacroIndicator,
    NewsArticle,
)
from qracer.data.registry import DataRegistry
from qracer.llm.providers import CompletionRequest, CompletionResponse, Role
from qracer.llm.registry import LLMRegistry
from qracer.models import Signal, SignalDirection

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLLM:
    """Fake LLM provider that returns canned responses."""

    def __init__(self, content: str = "[]") -> None:
        self.content = content
        self.calls: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        return CompletionResponse(
            content=self.content,
            model="fake",
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
        )


class FakePriceProvider:
    async def get_price(self, ticker: str) -> float:
        return 150.0

    async def get_ohlcv(self, ticker: str, start: date, end: date) -> list[OHLCV]:
        return [OHLCV(date=date(2024, 1, 1), open=100, high=105, low=99, close=102, volume=1000)]


class FakeFundamentalProvider:
    async def get_fundamentals(self, ticker: str) -> FundamentalData:
        return FundamentalData(
            ticker=ticker, pe_ratio=20.0, market_cap=1e12, revenue=1e10, earnings=1e9
        )


class FakeMacroProvider:
    async def get_indicator(self, name: str) -> MacroIndicator:
        return MacroIndicator(name=name, value=3.5, date=date(2024, 1, 1), source="test", unit="%")


class FakeNewsProvider:
    async def get_news(self, ticker: str, limit: int = 10) -> list[NewsArticle]:
        return [
            NewsArticle(
                title=f"{ticker} surges",
                source="TestNews",
                published_at=datetime(2024, 1, 1),
                url="https://example.com",
                summary="Stock went up.",
                sentiment=0.8,
            )
        ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_llm(content: str = "[]") -> tuple[LLMRegistry, FakeLLM]:
    registry = LLMRegistry()
    fake = FakeLLM(content)
    registry.register("fake", fake, [Role.RESEARCHER, Role.ANALYST, Role.STRATEGIST, Role.REPORTER])
    return registry, fake


def _make_data() -> DataRegistry:
    data = DataRegistry()
    from qracer.data.providers import (
        FundamentalProvider,
        MacroProvider,
        NewsProvider,
        PriceProvider,
    )

    data.register("price", FakePriceProvider(), [PriceProvider])
    data.register("fund", FakeFundamentalProvider(), [FundamentalProvider])
    data.register("macro", FakeMacroProvider(), [MacroProvider])
    data.register("news", FakeNewsProvider(), [NewsProvider])
    return data


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------


class TestBaseAgent:
    def test_is_abstract(self) -> None:
        assert issubclass(BaseAgent, BaseAgent)
        with pytest.raises(TypeError):
            BaseAgent(LLMRegistry(), DataRegistry())  # type: ignore[abstract]

    def test_successful_results_filters(self) -> None:
        from qracer.models import ToolResult

        results = [
            ToolResult(tool="a", success=True, data={"x": 1}, source="s"),
            ToolResult(tool="b", success=False, data={}, source="s", error="fail"),
            ToolResult(tool="c", success=True, data={"y": 2}, source="s"),
        ]
        filtered = BaseAgent._successful_results(results)
        assert len(filtered) == 2
        assert all(r.success for r in filtered)

    def test_format_tool_data(self) -> None:
        from qracer.models import ToolResult

        results = [
            ToolResult(tool="price", success=True, data={"ticker": "AAPL"}, source="P"),
            ToolResult(tool="fail", success=False, data={}, source="X", error="err"),
        ]
        text = BaseAgent._format_tool_data(results)
        assert "price" in text
        assert "AAPL" in text
        assert "fail" not in text


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------


class TestResearcher:
    def test_role(self) -> None:
        assert Researcher.role == Role.RESEARCHER

    async def test_screen_universe(self) -> None:
        llm, fake = _make_llm(json.dumps(["AAPL", "MSFT"]))
        data = _make_data()
        agent = Researcher(llm, data)

        stocks = await agent.screen_universe(["AAPL", "MSFT", "GOOG"])
        assert len(stocks) == 2
        assert stocks[0].ticker == "AAPL"
        assert stocks[1].ticker == "MSFT"
        assert len(fake.calls) == 1

    async def test_screen_universe_empty_on_no_data(self) -> None:
        llm, _ = _make_llm("[]")
        data = DataRegistry()  # no providers
        agent = Researcher(llm, data)

        stocks = await agent.screen_universe(["AAPL"])
        assert stocks == []

    async def test_screen_universe_market_cap_filter(self) -> None:
        llm, _ = _make_llm(json.dumps(["AAPL"]))
        data = _make_data()
        agent = Researcher(llm, data)

        # FakeFundamentalProvider returns market_cap=1e12, so 1e13 should filter all out
        stocks = await agent.screen_universe(["AAPL"], min_market_cap=1e13)
        # LLM still called but with empty data → returns empty
        assert stocks == []

    async def test_map_consensus(self) -> None:
        expected = {"sentiment": "bullish", "themes": ["AI"], "confidence": 8}
        llm, fake = _make_llm(json.dumps(expected))
        data = _make_data()
        agent = Researcher(llm, data)

        result = await agent.map_consensus("AAPL")
        assert result["sentiment"] == "bullish"
        assert len(fake.calls) == 1

    async def test_run_delegates_to_screen(self) -> None:
        llm, _ = _make_llm(json.dumps(["TSLA"]))
        data = _make_data()
        agent = Researcher(llm, data)

        stocks = await agent.run(["TSLA"])
        assert len(stocks) == 1
        assert stocks[0].ticker == "TSLA"


# ---------------------------------------------------------------------------
# Analyst
# ---------------------------------------------------------------------------


class TestAnalyst:
    def test_role(self) -> None:
        assert Analyst.role == Role.ANALYST

    async def test_detect_regime(self) -> None:
        expected = {"regime": "risk_on", "reasoning": "rates falling", "key_drivers": ["fed_rate"]}
        llm, fake = _make_llm(json.dumps(expected))
        data = _make_data()
        agent = Analyst(llm, data)

        result = await agent.detect_regime(["fed_rate", "cpi"])
        assert result["regime"] == "risk_on"
        assert len(fake.calls) == 1

    async def test_detect_regime_no_providers(self) -> None:
        llm, _ = _make_llm(json.dumps({"regime": "transition"}))
        data = DataRegistry()
        agent = Analyst(llm, data)

        result = await agent.detect_regime(["gdp"])
        assert "regime" in result

    async def test_discover_cross_market(self) -> None:
        expected = {
            "discoveries": [{"ticker": "AAPL", "signal": "bullish", "reasoning": "test"}],
            "causal_chains": ["Korea semi → US AI"],
        }
        llm, fake = _make_llm(json.dumps(expected))
        data = _make_data()
        agent = Analyst(llm, data)

        result = await agent.discover_cross_market(["AAPL", "005930.KS"])
        assert len(result["discoveries"]) == 1
        assert len(fake.calls) == 1

    async def test_run_delegates_to_cross_market(self) -> None:
        llm, _ = _make_llm(json.dumps({"discoveries": [], "causal_chains": []}))
        data = _make_data()
        agent = Analyst(llm, data)

        result = await agent.run(["AAPL"])
        assert "discoveries" in result


# ---------------------------------------------------------------------------
# Strategist
# ---------------------------------------------------------------------------


class TestStrategist:
    def test_role(self) -> None:
        assert Strategist.role == Role.STRATEGIST

    async def test_detect_contrarian(self) -> None:
        contrarian_signals = [
            {
                "ticker": "AAPL",
                "thesis": "oversold",
                "contrarian_angle": "market ignoring earnings beat",
                "direction": "long",
                "evidence": ["earnings up 20%"],
            }
        ]
        llm, fake = _make_llm(json.dumps(contrarian_signals))
        data = _make_data()
        agent = Strategist(llm, data)

        result = await agent.detect_contrarian(
            cross_market={"discoveries": []},
            consensus={"sentiment": "bearish"},
        )
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"

    async def test_detect_contrarian_bad_json(self) -> None:
        llm, _ = _make_llm("not json")
        data = _make_data()
        agent = Strategist(llm, data)

        result = await agent.detect_contrarian({}, {})
        assert result == []

    async def test_score_signals(self) -> None:
        scored = [
            {
                "ticker": "AAPL",
                "direction": "long",
                "conviction": 8.5,
                "thesis": "earnings beat",
                "evidence": ["Q4 beat"],
                "contrarian_angle": "ignored by market",
                "risk_factors": ["macro headwinds"],
                "time_horizon_days": 30,
            },
            {
                "ticker": "GOOG",
                "direction": "short",
                "conviction": 6.0,
                "thesis": "ad revenue declining",
                "evidence": [],
                "contrarian_angle": None,
                "risk_factors": [],
                "time_horizon_days": 60,
            },
        ]
        llm, _ = _make_llm(json.dumps(scored))
        data = _make_data()
        agent = Strategist(llm, data)

        signals = await agent.score_signals([{"ticker": "AAPL"}, {"ticker": "GOOG"}])
        assert len(signals) == 2
        assert signals[0].ticker == "AAPL"  # highest conviction first
        assert signals[0].conviction == 8.5
        assert signals[0].direction == SignalDirection.LONG
        assert signals[1].direction == SignalDirection.SHORT

    async def test_score_signals_empty(self) -> None:
        llm, fake = _make_llm("[]")
        data = _make_data()
        agent = Strategist(llm, data)

        signals = await agent.score_signals([])
        assert signals == []
        assert len(fake.calls) == 0  # should short-circuit

    async def test_run_full_pipeline(self) -> None:
        contrarian = [
            {
                "ticker": "TSLA",
                "thesis": "t",
                "contrarian_angle": "c",
                "direction": "long",
                "evidence": [],
            }
        ]
        scored = [
            {
                "ticker": "TSLA",
                "direction": "long",
                "conviction": 7.0,
                "thesis": "t",
                "evidence": [],
                "contrarian_angle": "c",
                "risk_factors": [],
                "time_horizon_days": 14,
            }
        ]
        # First call returns contrarian, second call returns scored
        responses = [json.dumps(contrarian), json.dumps(scored)]

        class MultiResponseLLM:
            def __init__(self):
                self.idx = 0

            async def complete(self, request: CompletionRequest) -> CompletionResponse:
                content = responses[self.idx]
                self.idx += 1
                return CompletionResponse(
                    content=content,
                    model="fake",
                    input_tokens=0,
                    output_tokens=0,
                    cost=0.0,
                )

        llm = LLMRegistry()
        llm.register("fake", MultiResponseLLM(), [Role.STRATEGIST])
        data = _make_data()
        agent = Strategist(llm, data)

        signals = await agent.run(cross_market={}, consensus={})
        assert len(signals) == 1
        assert signals[0].ticker == "TSLA"
        assert signals[0].conviction == 7.0


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


class TestReporter:
    def test_role(self) -> None:
        assert Reporter.role == Role.REPORTER

    async def test_generate_report_no_signals(self) -> None:
        llm, fake = _make_llm("{}")
        data = _make_data()
        agent = Reporter(llm, data)

        report = await agent.generate_report([])
        assert report.title == "No actionable signals"
        assert report.conviction == 0.0
        assert len(fake.calls) == 0  # should short-circuit

    async def test_generate_report(self) -> None:
        report_json = {
            "title": "AAPL: Hidden Earnings Catalyst",
            "what_happened": "Market missed the services growth inflection.",
            "evidence_chain": ["Services revenue +25% YoY — Source: Q4 10-Q"],
            "adversarial_check": ["Macro slowdown could compress multiples"],
            "verdict": "Buy with 8.5/10 conviction; risk-reward favourable.",
        }
        llm, fake = _make_llm(json.dumps(report_json))
        data = _make_data()
        agent = Reporter(llm, data)

        signal = Signal(
            ticker="AAPL",
            direction=SignalDirection.LONG,
            conviction=8.5,
            thesis="services growth",
            evidence=["services rev up"],
        )

        report = await agent.generate_report([signal])
        assert report.title == "AAPL: Hidden Earnings Catalyst"
        assert report.ticker == "AAPL"
        assert report.conviction == 8.5
        assert len(report.evidence_chain) == 1
        assert len(report.adversarial_check) == 1
        assert report.signals == [signal]
        assert len(fake.calls) == 1

    async def test_generate_report_bad_json_fallback(self) -> None:
        llm, _ = _make_llm("not valid json")
        data = _make_data()
        agent = Reporter(llm, data)

        signal = Signal(
            ticker="GOOG",
            direction=SignalDirection.SHORT,
            conviction=6.0,
            thesis="ad revenue down",
            evidence=["Q3 miss"],
            risk_factors=["regulation"],
        )

        report = await agent.generate_report([signal])
        # Falls back to signal data
        assert report.ticker == "GOOG"
        assert report.conviction == 6.0
        assert "GOOG" in report.title

    async def test_run_delegates_to_generate(self) -> None:
        report_json = {
            "title": "Test",
            "what_happened": "Test",
            "evidence_chain": [],
            "adversarial_check": [],
            "verdict": "Test",
        }
        llm, _ = _make_llm(json.dumps(report_json))
        data = _make_data()
        agent = Reporter(llm, data)

        signal = Signal(
            ticker="MSFT",
            direction=SignalDirection.LONG,
            conviction=7.0,
            thesis="cloud growth",
        )

        report = await agent.run([signal])
        assert report.title == "Test"
        assert report.ticker == "MSFT"
