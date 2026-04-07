"""Tests for FredAdapter."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from qracer.data.providers import MacroIndicator


class TestResolveSeriesMapping:
    """Test the indicator name -> FRED series ID resolution."""

    def test_known_indicators(self) -> None:
        from qracer.data.fred_adapter import _resolve_series

        assert _resolve_series("fed_funds_rate") == ("FEDFUNDS", "%")
        assert _resolve_series("fedfunds") == ("FEDFUNDS", "%")
        assert _resolve_series("10y_treasury") == ("DGS10", "%")
        assert _resolve_series("cpi") == ("CPIAUCSL", "index")
        assert _resolve_series("gdp") == ("GDP", "billions USD")
        assert _resolve_series("unemployment") == ("UNRATE", "%")
        assert _resolve_series("vix") == ("VIXCLS", "index")

    def test_case_insensitive(self) -> None:
        from qracer.data.fred_adapter import _resolve_series

        assert _resolve_series("FED_FUNDS_RATE") == ("FEDFUNDS", "%")
        assert _resolve_series("VIX") == ("VIXCLS", "index")
        assert _resolve_series("CPI") == ("CPIAUCSL", "index")

    def test_whitespace_normalized(self) -> None:
        from qracer.data.fred_adapter import _resolve_series

        assert _resolve_series("  fed funds rate  ") == ("FEDFUNDS", "%")

    def test_unknown_passes_through(self) -> None:
        from qracer.data.fred_adapter import _resolve_series

        assert _resolve_series("T10YIE") == ("T10YIE", "")


class TestFetchIndicator:
    """Test the synchronous _fetch_indicator helper."""

    def test_returns_macro_indicator(self) -> None:
        import pandas as pd

        from qracer.data.fred_adapter import _fetch_indicator

        mock_client = MagicMock()
        series = pd.Series(
            [5.25, 5.33, 5.33],
            index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        )
        mock_client.get_series.return_value = series

        result = _fetch_indicator(mock_client, "fed_funds_rate")

        mock_client.get_series.assert_called_once_with("FEDFUNDS")
        assert isinstance(result, MacroIndicator)
        assert result.name == "FEDFUNDS"
        assert result.value == 5.33
        assert result.date == date(2024, 3, 1)
        assert result.source == "fred"
        assert result.unit == "%"

    def test_empty_series_raises(self) -> None:
        import pandas as pd

        from qracer.data.fred_adapter import _fetch_indicator

        mock_client = MagicMock()
        mock_client.get_series.return_value = pd.Series(dtype=float)

        with pytest.raises(ValueError, match="No data available"):
            _fetch_indicator(mock_client, "gdp")

    def test_nan_values_dropped(self) -> None:
        import numpy as np
        import pandas as pd

        from qracer.data.fred_adapter import _fetch_indicator

        mock_client = MagicMock()
        series = pd.Series(
            [3.5, np.nan, np.nan],
            index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        )
        mock_client.get_series.return_value = series

        result = _fetch_indicator(mock_client, "unemployment")

        assert result.value == 3.5
        assert result.date == date(2024, 1, 1)

    def test_raw_series_id_passthrough(self) -> None:
        import pandas as pd

        from qracer.data.fred_adapter import _fetch_indicator

        mock_client = MagicMock()
        series = pd.Series([2.1], index=pd.to_datetime(["2024-06-01"]))
        mock_client.get_series.return_value = series

        result = _fetch_indicator(mock_client, "T10YIE")

        mock_client.get_series.assert_called_once_with("T10YIE")
        assert result.name == "T10YIE"
        assert result.unit == ""


class TestFredAdapterInit:
    """Test FredAdapter constructor validation."""

    @patch("qracer.data.fred_adapter._HAS_FRED", False)
    def test_missing_fredapi_raises(self) -> None:
        from qracer.data.fred_adapter import FredAdapter

        with pytest.raises(ImportError, match="fredapi is not installed"):
            FredAdapter(api_key="test")

    @patch("qracer.data.fred_adapter._HAS_FRED", True)
    def test_missing_api_key_raises(self) -> None:
        from qracer.data.fred_adapter import FredAdapter

        with pytest.raises(ValueError, match="FRED_API_KEY is required"):
            FredAdapter(api_key=None)

        with pytest.raises(ValueError, match="FRED_API_KEY is required"):
            FredAdapter(api_key="")

    @patch("qracer.data.fred_adapter._HAS_FRED", True)
    @patch("qracer.data.fred_adapter.Fred", create=True)
    def test_valid_init(self, mock_fred_cls: MagicMock) -> None:
        from qracer.data.fred_adapter import FredAdapter

        adapter = FredAdapter(api_key="test-key")
        mock_fred_cls.assert_called_once_with(api_key="test-key")
        assert adapter._client is mock_fred_cls.return_value


class TestFredAdapterAsync:
    """Test FredAdapter async method."""

    @patch("qracer.data.fred_adapter._HAS_FRED", True)
    @patch("qracer.data.fred_adapter.Fred", create=True)
    async def test_get_indicator(self, mock_fred_cls: MagicMock) -> None:
        import pandas as pd

        from qracer.data.fred_adapter import FredAdapter

        series = pd.Series([4.5], index=pd.to_datetime(["2024-05-01"]))
        mock_fred_cls.return_value.get_series.return_value = series

        adapter = FredAdapter(api_key="test-key")
        result = await adapter.get_indicator("vix")

        assert isinstance(result, MacroIndicator)
        assert result.name == "VIXCLS"
        assert result.value == 4.5
        assert result.source == "fred"
