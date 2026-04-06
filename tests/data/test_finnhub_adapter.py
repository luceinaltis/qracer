"""Tests for FinnhubAdapter API key validation."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

import qracer.data.finnhub_adapter as mod


@pytest.fixture()
def _has_finnhub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend finnhub-python is installed."""
    monkeypatch.setattr(mod, "_HAS_FINNHUB", True)
    # Inject a fake finnhub module so ``finnhub.Client()`` resolves
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "finnhub", fake)
    monkeypatch.setattr(mod, "finnhub", fake, raising=False)


class TestFinnhubAdapterInit:
    """FinnhubAdapter must reject missing or empty API keys at init time."""

    @pytest.mark.usefixtures("_has_finnhub")
    def test_none_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="requires an API key"):
            mod.FinnhubAdapter(api_key=None)

    @pytest.mark.usefixtures("_has_finnhub")
    def test_empty_string_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="requires an API key"):
            mod.FinnhubAdapter(api_key="")

    @pytest.mark.usefixtures("_has_finnhub")
    def test_no_args_raises(self) -> None:
        with pytest.raises(ValueError, match="requires an API key"):
            mod.FinnhubAdapter()

    @pytest.mark.usefixtures("_has_finnhub")
    def test_valid_api_key_accepted(self) -> None:
        adapter = mod.FinnhubAdapter(api_key="test-key-123")
        assert adapter._client is not None

    def test_missing_finnhub_package_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_HAS_FINNHUB", False)
        with pytest.raises(ImportError, match="finnhub-python is not installed"):
            mod.FinnhubAdapter(api_key="some-key")
