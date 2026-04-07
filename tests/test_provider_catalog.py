"""Tests for provider catalog plugin discovery."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from qracer.provider_catalog import (
    BUILTIN_DATA_PROVIDERS,
    BUILTIN_LLM_PROVIDERS,
    discover_data_providers,
    discover_llm_providers,
)


# ---------------------------------------------------------------------------
# Fake adapters used by the tests
# ---------------------------------------------------------------------------

class _FakePriceAdapter:
    """Satisfies PriceProvider protocol."""

    async def get_price(self, ticker: str) -> float:
        return 100.0

    async def get_ohlcv(self, ticker: str, start: date, end: date) -> list:
        return []


class _FakeNoCapAdapter:
    """Satisfies no known capability protocol."""

    def hello(self) -> str:
        return "hi"


class _FakeLLMAdapter:
    """Has a ``roles`` class attribute like built-in LLM adapters."""

    from qracer.llm.providers import Role

    roles = [Role.RESEARCHER, Role.ANALYST]


class _FakeLLMAdapterNoRoles:
    """Missing the ``roles`` attribute."""

    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry_point(name: str, cls: type) -> SimpleNamespace:
    """Create a minimal entry-point-like object."""
    return SimpleNamespace(name=name, load=lambda: cls)


def _make_failing_entry_point(name: str) -> SimpleNamespace:
    """Entry point whose load() raises ImportError."""

    def _boom():
        raise ImportError("missing dependency")

    return SimpleNamespace(name=name, load=_boom)


# ---------------------------------------------------------------------------
# discover_data_providers
# ---------------------------------------------------------------------------

class TestDiscoverDataProviders:
    def test_returns_builtins_when_no_plugins(self) -> None:
        result = discover_data_providers()
        assert result == BUILTIN_DATA_PROVIDERS

    def test_merges_external_plugin(self) -> None:
        ep = _make_entry_point("fake_price", _FakePriceAdapter)
        with patch(
            "qracer.provider_catalog.importlib.metadata.entry_points",
            return_value=[ep],
        ):
            result = discover_data_providers()

        # Built-ins preserved.
        for name in BUILTIN_DATA_PROVIDERS:
            assert name in result

        # External plugin added.
        assert "fake_price" in result
        adapter_path, caps = result["fake_price"]
        assert "PriceProvider" in adapter_path or "PriceProvider" in str(caps)
        assert any("PriceProvider" in c for c in caps)

    def test_skips_adapter_with_no_capabilities(self) -> None:
        ep = _make_entry_point("no_cap", _FakeNoCapAdapter)
        with patch(
            "qracer.provider_catalog.importlib.metadata.entry_points",
            return_value=[ep],
        ):
            result = discover_data_providers()

        assert "no_cap" not in result

    def test_skips_failing_entry_point(self) -> None:
        ep = _make_failing_entry_point("broken")
        with patch(
            "qracer.provider_catalog.importlib.metadata.entry_points",
            return_value=[ep],
        ):
            result = discover_data_providers()

        assert "broken" not in result
        # Built-ins still present.
        assert result == BUILTIN_DATA_PROVIDERS


# ---------------------------------------------------------------------------
# discover_llm_providers
# ---------------------------------------------------------------------------

class TestDiscoverLLMProviders:
    def test_returns_builtins_when_no_plugins(self) -> None:
        result = discover_llm_providers()
        assert result == BUILTIN_LLM_PROVIDERS

    def test_merges_external_plugin(self) -> None:
        ep = _make_entry_point("fake_llm", _FakeLLMAdapter)
        with patch(
            "qracer.provider_catalog.importlib.metadata.entry_points",
            return_value=[ep],
        ):
            result = discover_llm_providers()

        # Built-ins preserved.
        for name in BUILTIN_LLM_PROVIDERS:
            assert name in result

        # External plugin added with correct roles.
        assert "fake_llm" in result
        _, role_values = result["fake_llm"]
        assert "researcher" in role_values
        assert "analyst" in role_values

    def test_skips_adapter_without_roles(self) -> None:
        ep = _make_entry_point("no_roles", _FakeLLMAdapterNoRoles)
        with patch(
            "qracer.provider_catalog.importlib.metadata.entry_points",
            return_value=[ep],
        ):
            result = discover_llm_providers()

        assert "no_roles" not in result

    def test_skips_failing_entry_point(self) -> None:
        ep = _make_failing_entry_point("broken")
        with patch(
            "qracer.provider_catalog.importlib.metadata.entry_points",
            return_value=[ep],
        ):
            result = discover_llm_providers()

        assert "broken" not in result
        assert result == BUILTIN_LLM_PROVIDERS
