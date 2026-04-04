"""Tests for LLMRegistry."""

from __future__ import annotations

import pytest

from tracer.llm import LLMRegistry
from tracer.llm.providers import CompletionRequest, CompletionResponse, Role


class FakeLLMProvider:
    """Fake provider for testing."""

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(
            content="fake", model="fake-model", input_tokens=0, output_tokens=0, cost=0.0
        )


class FakeLLMProvider2:
    """Another fake provider (fallback)."""

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(
            content="fake2", model="fake-model-2", input_tokens=0, output_tokens=0, cost=0.0
        )


class TestLLMRegistry:
    def test_register_and_get(self) -> None:
        registry = LLMRegistry()
        provider = FakeLLMProvider()
        registry.register("fake", provider, [Role.RESEARCHER])
        assert registry.get(Role.RESEARCHER) is provider

    def test_get_by_name(self) -> None:
        registry = LLMRegistry()
        p1 = FakeLLMProvider()
        p2 = FakeLLMProvider2()
        registry.register("primary", p1, [Role.ANALYST])
        registry.register("fallback", p2, [Role.ANALYST])
        assert registry.get(Role.ANALYST, "fallback") is p2

    def test_priority_order(self) -> None:
        registry = LLMRegistry()
        p1 = FakeLLMProvider()
        p2 = FakeLLMProvider2()
        registry.register("primary", p1, [Role.STRATEGIST])
        registry.register("fallback", p2, [Role.STRATEGIST])
        assert registry.get(Role.STRATEGIST) is p1

    def test_get_all(self) -> None:
        registry = LLMRegistry()
        p1 = FakeLLMProvider()
        p2 = FakeLLMProvider2()
        registry.register("primary", p1, [Role.REPORTER])
        registry.register("fallback", p2, [Role.REPORTER])
        all_providers = registry.get_all(Role.REPORTER)
        assert len(all_providers) == 2
        assert all_providers[0] == ("primary", p1)
        assert all_providers[1] == ("fallback", p2)

    def test_missing_role_raises(self) -> None:
        registry = LLMRegistry()
        with pytest.raises(KeyError, match="No provider registered"):
            registry.get(Role.RESEARCHER)

    def test_missing_name_raises(self) -> None:
        registry = LLMRegistry()
        registry.register("fake", FakeLLMProvider(), [Role.RESEARCHER])
        with pytest.raises(KeyError, match="No provider named"):
            registry.get(Role.RESEARCHER, "nonexistent")

    def test_multiple_roles(self) -> None:
        registry = LLMRegistry()
        provider = FakeLLMProvider()
        registry.register("claude", provider, [Role.RESEARCHER, Role.ANALYST, Role.REPORTER])
        assert registry.get(Role.RESEARCHER) is provider
        assert registry.get(Role.ANALYST) is provider
        assert registry.get(Role.REPORTER) is provider
        assert len(registry.roles()) == 3

    def test_get_all_empty_role(self) -> None:
        registry = LLMRegistry()
        assert registry.get_all(Role.STRATEGIST) == []
