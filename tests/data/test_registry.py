"""Tests for DataRegistry."""

from __future__ import annotations

from datetime import date

import pytest

from qracer.data import DataRegistry, PriceProvider
from qracer.data.providers import (
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


class AsyncFailingPriceAdapter:
    """Async adapter that always raises."""

    async def get_price(self, ticker: str) -> float:
        raise RuntimeError("primary down")

    async def get_ohlcv(self, ticker: str, start: date, end: date) -> list[OHLCV]:
        raise RuntimeError("primary down")


class AsyncSucceedingPriceAdapter:
    """Async adapter that returns a fixed price."""

    async def get_price(self, ticker: str) -> float:
        return 42.0

    async def get_ohlcv(self, ticker: str, start: date, end: date) -> list[OHLCV]:
        return []


class TestAsyncGetWithFallback:
    async def test_first_adapter_succeeds(self) -> None:
        registry = DataRegistry()
        registry.register("good", AsyncSucceedingPriceAdapter(), [PriceProvider])
        registry.register("also_good", AsyncSucceedingPriceAdapter(), [PriceProvider])
        result = await registry.async_get_with_fallback(PriceProvider, "get_price", "AAPL")
        assert result == 42.0

    async def test_first_fails_second_succeeds(self) -> None:
        registry = DataRegistry()
        registry.register("bad", AsyncFailingPriceAdapter(), [PriceProvider])
        registry.register("good", AsyncSucceedingPriceAdapter(), [PriceProvider])
        result = await registry.async_get_with_fallback(PriceProvider, "get_price", "AAPL")
        assert result == 42.0

    async def test_all_fail_raises_last(self) -> None:
        registry = DataRegistry()
        registry.register("bad1", AsyncFailingPriceAdapter(), [PriceProvider])
        registry.register("bad2", AsyncFailingPriceAdapter(), [PriceProvider])
        with pytest.raises(RuntimeError, match="primary down"):
            await registry.async_get_with_fallback(PriceProvider, "get_price", "AAPL")

    async def test_missing_capability_raises(self) -> None:
        registry = DataRegistry()
        with pytest.raises(KeyError, match="No adapter registered"):
            await registry.async_get_with_fallback(PriceProvider, "get_price", "AAPL")

    async def test_kwargs_forwarded(self) -> None:
        class AsyncKwargAdapter:
            async def fetch(self, ticker: str, period: str = "1d") -> str:
                return f"{ticker}-{period}"

        registry = DataRegistry()
        registry.register("kw", AsyncKwargAdapter(), [PriceProvider])
        result = await registry.async_get_with_fallback(PriceProvider, "fetch", "AAPL", period="5d")
        assert result == "AAPL-5d"

    async def test_ohlcv_fallback(self) -> None:
        """Verify fallback works for get_ohlcv with positional args."""
        registry = DataRegistry()
        registry.register("bad", AsyncFailingPriceAdapter(), [PriceProvider])
        registry.register("good", AsyncSucceedingPriceAdapter(), [PriceProvider])
        result = await registry.async_get_with_fallback(
            PriceProvider, "get_ohlcv", "AAPL", date.today(), date.today()
        )
        assert result == []


class FailingPriceAdapter:
    """Adapter that always raises on get_price."""

    def get_price(self, ticker: str) -> float:
        raise RuntimeError("primary down")

    def get_ohlcv(self, ticker: str, start: date, end: date) -> list[OHLCV]:
        raise RuntimeError("primary down")


class SucceedingPriceAdapter:
    """Adapter that returns a fixed price synchronously."""

    def get_price(self, ticker: str) -> float:
        return 42.0

    def get_ohlcv(self, ticker: str, start: date, end: date) -> list[OHLCV]:
        return []


class TestGetWithFallback:
    def test_first_adapter_succeeds(self) -> None:
        registry = DataRegistry()
        registry.register("good", SucceedingPriceAdapter(), [PriceProvider])
        registry.register("also_good", SucceedingPriceAdapter(), [PriceProvider])
        result = registry.get_with_fallback(PriceProvider, "get_price", "AAPL")
        assert result == 42.0

    def test_first_fails_second_succeeds(self) -> None:
        registry = DataRegistry()
        registry.register("bad", FailingPriceAdapter(), [PriceProvider])
        registry.register("good", SucceedingPriceAdapter(), [PriceProvider])
        result = registry.get_with_fallback(PriceProvider, "get_price", "AAPL")
        assert result == 42.0

    def test_all_fail_raises_last(self) -> None:
        registry = DataRegistry()
        registry.register("bad1", FailingPriceAdapter(), [PriceProvider])
        registry.register("bad2", FailingPriceAdapter(), [PriceProvider])
        with pytest.raises(RuntimeError, match="primary down"):
            registry.get_with_fallback(PriceProvider, "get_price", "AAPL")

    def test_missing_capability_raises(self) -> None:
        registry = DataRegistry()
        with pytest.raises(KeyError, match="No adapter registered"):
            registry.get_with_fallback(PriceProvider, "get_price", "AAPL")

    def test_kwargs_forwarded(self) -> None:
        class KwargAdapter:
            def fetch(self, ticker: str, period: str = "1d") -> str:
                return f"{ticker}-{period}"

        registry = DataRegistry()
        registry.register("kw", KwargAdapter(), [PriceProvider])
        result = registry.get_with_fallback(PriceProvider, "fetch", "AAPL", period="5d")
        assert result == "AAPL-5d"
