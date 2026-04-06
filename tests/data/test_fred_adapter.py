"""Tests for FredAdapter — FRED macroeconomic data adapter."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from qracer.data.providers import MacroIndicator, MacroProvider


# ---------------------------------------------------------------------------
# Import guard — tests work even when fredapi is not installed
# ---------------------------------------------------------------------------

@pytest.fixture()
def _mock_fredapi(monkeypatch: pytest.MonkeyPatch):
    """Patch the fredapi import so FredAdapter can be instantiated."""
    fake_fred_mod = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "fredapi", fake_fred_mod)

    import qracer.data.fred_adapter as mod

    monkeypatch.setattr(mod, "_HAS_FRED", True)
    # Fred is only defined when fredapi is installed; inject it into the module.
    if not hasattr(mod, "Fred"):
        monkeypatch.setattr(mod, "Fred", fake_fred_mod.Fred, raising=False)
    else:
        monkeypatch.setattr(mod, "Fred", fake_fred_mod.Fred)
    return fake_fred_mod


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------

class TestFredAdapterInit:
    def test_raises_without_fredapi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import qracer.data.fred_adapter as mod

        monkeypatch.setattr(mod, "_HAS_FRED", False)
        with pytest.raises(ImportError, match="fredapi is not installed"):
            mod.FredAdapter(api_key="test-key")

    def test_raises_without_api_key(self, _mock_fredapi: MagicMock) -> None:
        from qracer.data.fred_adapter import FredAdapter

        with pytest.raises(ValueError, match="FRED API key is required"):
            FredAdapter(api_key=None)

    def test_raises_with_empty_api_key(self, _mock_fredapi: MagicMock) -> None:
        from qracer.data.fred_adapter import FredAdapter

        with pytest.raises(ValueError, match="FRED API key is required"):
            FredAdapter(api_key="")

    def test_success_with_api_key(self, _mock_fredapi: MagicMock) -> None:
        from qracer.data.fred_adapter import FredAdapter

        adapter = FredAdapter(api_key="my-key")
        assert adapter is not None


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocolConformance:
    def test_satisfies_macro_provider(self, _mock_fredapi: MagicMock) -> None:
        from qracer.data.fred_adapter import FredAdapter

        adapter = FredAdapter(api_key="test-key")
        assert isinstance(adapter, MacroProvider)


# ---------------------------------------------------------------------------
# get_indicator tests
# ---------------------------------------------------------------------------

class TestGetIndicator:
    async def test_known_indicator(self, _mock_fredapi: MagicMock) -> None:
        from qracer.data.fred_adapter import FredAdapter

        series = pd.Series(
            [5.25, 5.33],
            index=pd.to_datetime(["2024-01-01", "2024-02-01"]),
        )
        mock_client = MagicMock()
        mock_client.get_series.return_value = series

        adapter = FredAdapter(api_key="test-key")
        adapter._client = mock_client

        result = await adapter.get_indicator("fed_funds_rate")

        mock_client.get_series.assert_called_once_with("FEDFUNDS")
        assert isinstance(result, MacroIndicator)
        assert result.name == "fed_funds_rate"
        assert result.value == 5.33
        assert result.date == date(2024, 2, 1)
        assert result.source == "FRED"
        assert result.unit == "%"

    async def test_direct_series_id_fallback(self, _mock_fredapi: MagicMock) -> None:
        from qracer.data.fred_adapter import FredAdapter

        series = pd.Series(
            [100.5],
            index=pd.to_datetime(["2024-03-01"]),
        )
        mock_client = MagicMock()
        mock_client.get_series.return_value = series

        adapter = FredAdapter(api_key="test-key")
        adapter._client = mock_client

        result = await adapter.get_indicator("CUSTOM_SERIES")

        mock_client.get_series.assert_called_once_with("CUSTOM_SERIES")
        assert result.value == 100.5
        assert result.unit == ""

    async def test_empty_series_raises(self, _mock_fredapi: MagicMock) -> None:
        from qracer.data.fred_adapter import FredAdapter

        mock_client = MagicMock()
        mock_client.get_series.return_value = pd.Series(dtype=float)

        adapter = FredAdapter(api_key="test-key")
        adapter._client = mock_client

        with pytest.raises(RuntimeError, match="No data returned"):
            await adapter.get_indicator("fed_funds_rate")

    async def test_none_series_raises(self, _mock_fredapi: MagicMock) -> None:
        from qracer.data.fred_adapter import FredAdapter

        mock_client = MagicMock()
        mock_client.get_series.return_value = None

        adapter = FredAdapter(api_key="test-key")
        adapter._client = mock_client

        with pytest.raises(RuntimeError, match="No data returned"):
            await adapter.get_indicator("vix")

    async def test_nan_values_skipped(self, _mock_fredapi: MagicMock) -> None:
        from qracer.data.fred_adapter import FredAdapter

        series = pd.Series(
            [3.5, float("nan")],
            index=pd.to_datetime(["2024-01-01", "2024-02-01"]),
        )
        mock_client = MagicMock()
        mock_client.get_series.return_value = series

        adapter = FredAdapter(api_key="test-key")
        adapter._client = mock_client

        result = await adapter.get_indicator("unemployment")

        assert result.value == 3.5
        assert result.date == date(2024, 1, 1)

    async def test_all_known_indicators_mapped(self, _mock_fredapi: MagicMock) -> None:
        """Verify all six required indicators have mappings."""
        from qracer.data.fred_adapter import _INDICATOR_MAP

        expected = {"fed_funds_rate", "treasury_10y", "cpi_yoy", "gdp_growth", "unemployment", "vix"}
        assert set(_INDICATOR_MAP.keys()) == expected


# ---------------------------------------------------------------------------
# Provider catalog integration
# ---------------------------------------------------------------------------

class TestCatalogRegistration:
    def test_fred_in_catalog(self) -> None:
        from qracer.provider_catalog import BUILTIN_DATA_PROVIDERS

        assert "fred" in BUILTIN_DATA_PROVIDERS
        path, caps = BUILTIN_DATA_PROVIDERS["fred"]
        assert path == "qracer.data.fred_adapter.FredAdapter"
        assert "qracer.data.providers.MacroProvider" in caps
