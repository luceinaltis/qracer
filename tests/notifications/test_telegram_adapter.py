"""Tests for TelegramAdapter."""

import json
from unittest.mock import MagicMock, patch

import pytest

from qracer.notifications.providers import Notification, NotificationCategory
from qracer.notifications.telegram_adapter import TelegramAdapter, _format_message


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestTelegramAdapterInit:
    def test_requires_bot_token(self):
        with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
            TelegramAdapter(bot_token="", chat_id="12345")

    def test_requires_chat_id(self):
        with pytest.raises(ValueError, match="TELEGRAM_CHAT_ID"):
            TelegramAdapter(bot_token="tok123", chat_id="")

    def test_valid_construction(self):
        adapter = TelegramAdapter(bot_token="tok123", chat_id="12345")
        assert adapter._bot_token == "tok123"
        assert adapter._chat_id == "12345"


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------


class TestFormatMessage:
    def test_format_includes_category_and_title(self):
        n = Notification(
            category=NotificationCategory.PRICE_ALERT,
            title="AAPL above $200",
            body="Current price: $201.50",
        )
        msg = _format_message(n)
        assert "*[Price Alert]*" in msg
        assert "*AAPL above $200*" in msg
        assert "Current price: $201.50" in msg

    def test_format_portfolio_drawdown(self):
        n = Notification(
            category=NotificationCategory.PORTFOLIO_DRAWDOWN,
            title="Drawdown warning",
            body="Down 12%",
        )
        msg = _format_message(n)
        assert "*[Portfolio Drawdown]*" in msg


# ---------------------------------------------------------------------------
# Send — mocked HTTP
# ---------------------------------------------------------------------------


def _make_notification() -> Notification:
    return Notification(
        category=NotificationCategory.RESEARCH_COMPLETE,
        title="Analysis done",
        body="AAPL deep dive finished.",
    )


class TestTelegramSend:
    @pytest.fixture()
    def adapter(self) -> TelegramAdapter:
        return TelegramAdapter(bot_token="test-token", chat_id="999")

    async def test_send_success(self, adapter: TelegramAdapter):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("qracer.notifications.telegram_adapter.urllib.request.urlopen", return_value=mock_resp) as mock_open:
            result = await adapter.send(_make_notification())

        assert result is True
        call_args = mock_open.call_args
        req = call_args[0][0]
        assert "test-token" in req.full_url
        body = json.loads(req.data)
        assert body["chat_id"] == "999"
        assert body["parse_mode"] == "Markdown"

    async def test_send_http_error(self, adapter: TelegramAdapter):
        import urllib.error

        exc = urllib.error.HTTPError(
            url="https://api.telegram.org", code=401, msg="Unauthorized", hdrs=None, fp=None  # type: ignore[arg-type]
        )
        with patch("qracer.notifications.telegram_adapter.urllib.request.urlopen", side_effect=exc):
            result = await adapter.send(_make_notification())

        assert result is False

    async def test_send_url_error(self, adapter: TelegramAdapter):
        import urllib.error

        exc = urllib.error.URLError(reason="Connection refused")
        with patch("qracer.notifications.telegram_adapter.urllib.request.urlopen", side_effect=exc):
            result = await adapter.send(_make_notification())

        assert result is False

    async def test_send_non_200_returns_false(self, adapter: TelegramAdapter):
        mock_resp = MagicMock()
        mock_resp.status = 400
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("qracer.notifications.telegram_adapter.urllib.request.urlopen", return_value=mock_resp):
            result = await adapter.send(_make_notification())

        assert result is False
