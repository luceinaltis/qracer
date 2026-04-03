"""Tests for DataRegistry."""

from __future__ import annotations

from datetime import date

import pytest

from tracer.data import DataRegistry, PriceProvider
from tracer.data.providers import (
    OHLCV,
    FundamentalData,
    FundamentalProvider,
    MacroIndicator,
    MacroProvider,
)


class FakePriceAdapter:
    """Fake adapter implementing PriceProvider."""

    async def get_price(self, ticker: str) -> float:
        return 100.0

    async def get_ohlcv(self, ticker: str, start: date, end: date) -> list[OHLCV]:
        return []


class FakePriceAdapter2:
    """Another fake price adapter (fallback)."""

    async def get_price(self, ticker: str) -> float:
        return 200.0

    async def get_ohlcv(self, ticker: str, start: date, end: date) -> list[OHLCV]:
        return []


class FakeFundamentalAdapter:
    """Fake adapter implementing FundamentalProvider."""

    async def get_fundamentals(self, ticker: str) -> FundamentalData:
        return FundamentalData(ticker=ticker, pe_ratio=15.0)


class FakeMacroAdapter:
    """Fake adapter implementing MacroProvider."""

    async def get_indicator(self, name: str) -> MacroIndicator:
        return MacroIndicator(name=name, value=2.5, date=date.today(), source="test")


class TestDataRegistry:
    def test_register_and_get(self) -> None:
        registry = DataRegistry()
        adapter = FakePriceAdapter()
        registry.register("fake", adapter, [PriceProvider])
        assert registry.get(PriceProvider) is adapter

    def test_get_by_name(self) -> None:
        registry = DataRegistry()
        a1 = FakePriceAdapter()
        a2 = FakePriceAdapter2()
        registry.register("primary", a1, [PriceProvider])
        registry.register("fallback", a2, [PriceProvider])
        assert registry.get(PriceProvider, "fallback") is a2

    def test_priority_order(self) -> None:
        registry = DataRegistry()
        a1 = FakePriceAdapter()
        a2 = FakePriceAdapter2()
        registry.register("primary", a1, [PriceProvider])
        registry.register("fallback", a2, [PriceProvider])
        # Default returns highest priority (first registered)
        assert registry.get(PriceProvider) is a1

    def test_get_all(self) -> None:
        registry = DataRegistry()
        a1 = FakePriceAdapter()
        a2 = FakePriceAdapter2()
        registry.register("primary", a1, [PriceProvider])
        registry.register("fallback", a2, [PriceProvider])
        all_adapters = registry.get_all(PriceProvider)
        assert len(all_adapters) == 2
        assert all_adapters[0] == ("primary", a1)
        assert all_adapters[1] == ("fallback", a2)

    def test_missing_capability_raises(self) -> None:
        registry = DataRegistry()
        with pytest.raises(KeyError, match="No adapter registered"):
            registry.get(PriceProvider)

    def test_missing_name_raises(self) -> None:
        registry = DataRegistry()
        registry.register("fake", FakePriceAdapter(), [PriceProvider])
        with pytest.raises(KeyError, match="No adapter named"):
            registry.get(PriceProvider, "nonexistent")

    def test_multiple_capabilities(self) -> None:
        registry = DataRegistry()
        price = FakePriceAdapter()
        fundamental = FakeFundamentalAdapter()
        macro = FakeMacroAdapter()
        registry.register("price", price, [PriceProvider])
        registry.register("fundamental", fundamental, [FundamentalProvider])
        registry.register("macro", macro, [MacroProvider])
        assert registry.get(PriceProvider) is price
        assert registry.get(FundamentalProvider) is fundamental
        assert registry.get(MacroProvider) is macro
        assert len(registry.capabilities()) == 3

    def test_get_all_empty_capability(self) -> None:
        registry = DataRegistry()
        assert registry.get_all(PriceProvider) == []
