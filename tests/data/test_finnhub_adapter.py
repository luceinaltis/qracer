"""Tests for FinnhubAdapter API key validation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestFinnhubAdapterInit:
    """Test FinnhubAdapter constructor validation."""

    @patch("qracer.data.finnhub_adapter._HAS_FINNHUB", False)
    def test_missing_finnhub_raises(self) -> None:
        from qracer.data.finnhub_adapter import FinnhubAdapter

        with pytest.raises(ImportError, match="finnhub-python is not installed"):
            FinnhubAdapter(api_key="test")

    @patch("qracer.data.finnhub_adapter._HAS_FINNHUB", True)
    def test_missing_api_key_raises(self) -> None:
        from qracer.data.finnhub_adapter import FinnhubAdapter

        with pytest.raises(ValueError, match="FINNHUB_API_KEY is required"):
            FinnhubAdapter(api_key=None)

    @patch("qracer.data.finnhub_adapter._HAS_FINNHUB", True)
    def test_empty_api_key_raises(self) -> None:
        from qracer.data.finnhub_adapter import FinnhubAdapter

        with pytest.raises(ValueError, match="FINNHUB_API_KEY is required"):
            FinnhubAdapter(api_key="")

    @patch("qracer.data.finnhub_adapter._HAS_FINNHUB", True)
    @patch("qracer.data.finnhub_adapter.finnhub", create=True)
    def test_valid_api_key_succeeds(self, mock_finnhub: MagicMock) -> None:
        from qracer.data.finnhub_adapter import FinnhubAdapter

        adapter = FinnhubAdapter(api_key="valid-key")
        mock_finnhub.Client.assert_called_once_with(api_key="valid-key")
        assert adapter._client is mock_finnhub.Client.return_value
