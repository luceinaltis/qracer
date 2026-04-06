"""Shared test fixtures and fake objects.

Centralises fakes, factory functions, and pytest fixtures used across
multiple test modules.  Import fakes directly; fixtures are discovered
by pytest automatically.
"""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import AsyncMock

from tracer.data.providers import (
    OHLCV,
    FundamentalData,
    MacroIndicator,
    NewsArticle,
)
from tracer.data.registry import DataRegistry
from tracer.llm.providers import CompletionRequest, CompletionResponse, Role
from tracer.llm.registry import LLMRegistry
from tracer.memory.session_logger import TurnRecord
from tracer.models import ToolResult

# ---------------------------------------------------------------------------
# Fake LLM providers
# ---------------------------------------------------------------------------


class FakeLLM:
    """Fake LLM provider that returns canned responses.

    Tracks all calls for assertion in tests.
    """

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


# ---------------------------------------------------------------------------
# Fake data providers
# ---------------------------------------------------------------------------


class FakePriceProvider:
    """Async fake implementing PriceProvider protocol."""

    async def get_price(self, ticker: str) -> float:
        return 150.0

    async def get_ohlcv(self, ticker: str, start: date, end: date) -> list[OHLCV]:
        return [OHLCV(date=date(2024, 1, 1), open=100, high=105, low=99, close=102, volume=1000)]


class FakeFundamentalProvider:
    """Async fake implementing FundamentalProvider protocol."""

    async def get_fundamentals(self, ticker: str) -> FundamentalData:
        return FundamentalData(
            ticker=ticker, pe_ratio=20.0, market_cap=1e12, revenue=1e10, earnings=1e9
        )


class FakeMacroProvider:
    """Async fake implementing MacroProvider protocol."""

    async def get_indicator(self, name: str) -> MacroIndicator:
        return MacroIndicator(name=name, value=3.5, date=date(2024, 1, 1), source="test", unit="%")


class FakeNewsProvider:
    """Async fake implementing NewsProvider protocol."""

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
# Factory functions
# ---------------------------------------------------------------------------


def make_llm_registry(content: str = "[]") -> tuple[LLMRegistry, FakeLLM]:
    """Create an LLMRegistry with a FakeLLM registered for all roles."""
    registry = LLMRegistry()
    fake = FakeLLM(content)
    registry.register("fake", fake, [Role.RESEARCHER, Role.ANALYST, Role.STRATEGIST, Role.REPORTER])
    return registry, fake


def make_data_registry() -> DataRegistry:
    """Create a DataRegistry with all fake providers registered."""
    from tracer.data.providers import (
        FundamentalProvider,
        MacroProvider,
        NewsProvider,
        PriceProvider,
    )

    data = DataRegistry()
    data.register("price", FakePriceProvider(), [PriceProvider])
    data.register("fund", FakeFundamentalProvider(), [FundamentalProvider])
    data.register("macro", FakeMacroProvider(), [MacroProvider])
    data.register("news", FakeNewsProvider(), [NewsProvider])
    return data


def make_mock_llm_registry(responses: dict[Role, str | list[str]]) -> LLMRegistry:
    """Build an LLMRegistry with AsyncMock providers returning canned responses.

    If a role maps to a list, responses are returned in sequence.
    """
    registry = LLMRegistry()
    for role, content in responses.items():
        provider = AsyncMock()
        if isinstance(content, list):
            contents = list(content)

            async def _complete(req, _contents=contents):  # type: ignore[no-untyped-def]
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


def make_single_role_registry(role: Role, response_content: str) -> LLMRegistry:
    """Build an LLMRegistry with one mock provider for a single role."""
    mock_provider = AsyncMock()
    mock_provider.complete.return_value = CompletionResponse(
        content=response_content,
        model="mock",
        input_tokens=10,
        output_tokens=5,
        cost=0.0,
    )
    registry = LLMRegistry()
    registry.register("mock", mock_provider, [role])
    return registry


# ---------------------------------------------------------------------------
# ToolResult helpers
# ---------------------------------------------------------------------------


def ok_result(tool: str, data: dict | None = None) -> ToolResult:
    """Create a successful ToolResult for testing."""
    return ToolResult(
        tool=tool,
        success=True,
        data=data or {"sample": "data"},
        source="test",
        fetched_at=datetime.now(),
        is_stale=False,
    )


def failed_result(tool: str) -> ToolResult:
    """Create a failed ToolResult for testing."""
    return ToolResult(
        tool=tool,
        success=False,
        data={},
        source="test",
        error="test error",
    )


def sample_analysis_results() -> list[ToolResult]:
    """Create sample analysis results for trade thesis tests."""
    return [
        ToolResult(
            tool="price_event",
            success=True,
            data={"ticker": "AAPL", "current_price": 185.0},
            source="PriceProvider",
        ),
        ToolResult(
            tool="fundamentals",
            success=True,
            data={"ticker": "AAPL", "pe_ratio": 28.5},
            source="FundamentalProvider",
        ),
    ]


# ---------------------------------------------------------------------------
# TurnRecord helper
# ---------------------------------------------------------------------------


def make_turn(content: str, turn: int = 1, role: str = "user", ts: str | None = None) -> TurnRecord:
    """Build a TurnRecord for context tests."""
    return TurnRecord(
        turn=turn,
        role=role,
        content=content,
        ts=ts or datetime.now().isoformat(),
    )
