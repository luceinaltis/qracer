"""Tests for TelegramBotPoller and BotCommand."""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from qracer.notifications.telegram_poller import (
    BotCommand,
    TelegramBotPoller,
)

# ---------------------------------------------------------------------------
# BotCommand.parse
# ---------------------------------------------------------------------------


class TestBotCommandParse:
    def test_simple_command(self) -> None:
        cmd = BotCommand.parse("/status")
        assert cmd is not None
        assert cmd.action == "status"
        assert cmd.args == []
        assert cmd.raw_text == "/status"

    def test_command_with_args(self) -> None:
        cmd = BotCommand.parse("/analyze AAPL")
        assert cmd is not None
        assert cmd.action == "analyze"
        assert cmd.args == ["AAPL"]

    def test_command_with_multiple_args(self) -> None:
        cmd = BotCommand.parse("/alert AAPL above 200")
        assert cmd is not None
        assert cmd.action == "alert"
        assert cmd.args == ["AAPL", "above", "200"]

    def test_command_with_botname_suffix(self) -> None:
        cmd = BotCommand.parse("/analyze@qracerbot AAPL")
        assert cmd is not None
        assert cmd.action == "analyze"
        assert cmd.args == ["AAPL"]

    def test_action_lowercased(self) -> None:
        cmd = BotCommand.parse("/STATUS")
        assert cmd is not None
        assert cmd.action == "status"

    def test_leading_trailing_whitespace(self) -> None:
        cmd = BotCommand.parse("   /tasks   ")
        assert cmd is not None
        assert cmd.action == "tasks"

    def test_non_command_returns_none(self) -> None:
        assert BotCommand.parse("hello world") is None

    def test_empty_string_returns_none(self) -> None:
        assert BotCommand.parse("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert BotCommand.parse("   ") is None

    def test_lone_slash_returns_none(self) -> None:
        assert BotCommand.parse("/") is None

    def test_only_botname_after_slash_returns_none(self) -> None:
        assert BotCommand.parse("/@bot") is None


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestTelegramBotPollerInit:
    def test_requires_bot_token(self) -> None:
        with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
            TelegramBotPoller(bot_token="", chat_id="12345")

    def test_requires_chat_id(self) -> None:
        with pytest.raises(ValueError, match="TELEGRAM_CHAT_ID"):
            TelegramBotPoller(bot_token="tok", chat_id="")

    def test_chat_id_coerced_to_str(self) -> None:
        poller = TelegramBotPoller(bot_token="tok", chat_id=12345)  # type: ignore[arg-type]
        assert poller.chat_id == "12345"

    def test_initial_offset_is_none(self) -> None:
        poller = TelegramBotPoller(bot_token="tok", chat_id="1")
        assert poller.offset is None


# ---------------------------------------------------------------------------
# poll() — mocked HTTP
# ---------------------------------------------------------------------------


def _mock_response(payload: dict) -> MagicMock:
    """Build a context-manager mock that ``urlopen`` returns."""
    body = json.dumps(payload).encode()
    resp = MagicMock()
    resp.status = 200
    resp.read.return_value = body
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _make_update(update_id: int, chat_id: int, text: str) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "chat": {"id": chat_id, "type": "private"},
            "text": text,
        },
    }


class TestTelegramBotPollerPoll:
    @pytest.fixture()
    def poller(self) -> TelegramBotPoller:
        return TelegramBotPoller(bot_token="tok", chat_id="999", timeout=1)

    async def test_poll_parses_commands_and_advances_offset(
        self, poller: TelegramBotPoller
    ) -> None:
        payload = {
            "ok": True,
            "result": [
                _make_update(10, 999, "/status"),
                _make_update(11, 999, "/analyze AAPL"),
            ],
        }
        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        with patch(target, return_value=_mock_response(payload)):
            commands = await poller.poll()

        assert len(commands) == 2
        assert commands[0].action == "status"
        assert commands[1].action == "analyze"
        assert commands[1].args == ["AAPL"]
        assert poller.offset == 12  # max update_id + 1

    async def test_poll_filters_unauthorised_chats(self, poller: TelegramBotPoller) -> None:
        payload = {
            "ok": True,
            "result": [
                _make_update(1, 999, "/status"),
                _make_update(2, 1234, "/leak"),  # different chat — must be ignored
                _make_update(3, 999, "/tasks"),
            ],
        }
        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        with patch(target, return_value=_mock_response(payload)):
            commands = await poller.poll()

        actions = [c.action for c in commands]
        assert actions == ["status", "tasks"]
        # Offset still advances past all updates so we don't re-fetch them.
        assert poller.offset == 4

    async def test_poll_skips_non_text_messages(self, poller: TelegramBotPoller) -> None:
        payload = {
            "ok": True,
            "result": [
                {
                    "update_id": 5,
                    "message": {
                        "message_id": 5,
                        "chat": {"id": 999, "type": "private"},
                        "photo": [{"file_id": "abc"}],
                    },
                },
                _make_update(6, 999, "/status"),
            ],
        }
        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        with patch(target, return_value=_mock_response(payload)):
            commands = await poller.poll()

        assert len(commands) == 1
        assert commands[0].action == "status"
        assert poller.offset == 7

    async def test_poll_skips_non_command_text(self, poller: TelegramBotPoller) -> None:
        payload = {
            "ok": True,
            "result": [_make_update(1, 999, "hello there")],
        }
        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        with patch(target, return_value=_mock_response(payload)):
            commands = await poller.poll()

        assert commands == []
        assert poller.offset == 2  # update_id still consumed

    async def test_poll_passes_offset_on_subsequent_calls(self, poller: TelegramBotPoller) -> None:
        payload1 = {"ok": True, "result": [_make_update(20, 999, "/status")]}
        payload2 = {"ok": True, "result": [_make_update(21, 999, "/tasks")]}

        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        responses = [_mock_response(payload1), _mock_response(payload2)]
        with patch(target, side_effect=responses) as mock:
            await poller.poll()
            await poller.poll()

        # First call has no offset, second call must include offset=21.
        first_url = mock.call_args_list[0][0][0]
        second_url = mock.call_args_list[1][0][0]
        assert "offset=" not in first_url
        assert "offset=21" in second_url

    async def test_poll_returns_empty_on_http_error(self, poller: TelegramBotPoller) -> None:
        exc = urllib.error.HTTPError(
            url="https://api.telegram.org",
            code=500,
            msg="Server Error",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )
        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        with patch(target, side_effect=exc):
            commands = await poller.poll()

        assert commands == []
        assert poller.offset is None

    async def test_poll_returns_empty_on_url_error(self, poller: TelegramBotPoller) -> None:
        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        with patch(target, side_effect=urllib.error.URLError("Connection refused")):
            commands = await poller.poll()
        assert commands == []

    async def test_poll_returns_empty_on_invalid_json(self, poller: TelegramBotPoller) -> None:
        resp = MagicMock()
        resp.status = 200
        resp.read.return_value = b"<not json>"
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        with patch(target, return_value=resp):
            commands = await poller.poll()
        assert commands == []

    async def test_poll_returns_empty_on_ok_false(self, poller: TelegramBotPoller) -> None:
        payload = {"ok": False, "description": "bad token"}
        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        with patch(target, return_value=_mock_response(payload)):
            commands = await poller.poll()
        assert commands == []

    async def test_poll_uses_token_in_url(self, poller: TelegramBotPoller) -> None:
        payload = {"ok": True, "result": []}
        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        with patch(target, return_value=_mock_response(payload)) as mock:
            await poller.poll()
        url = mock.call_args[0][0]
        assert "/bottok/getUpdates" in url


# ---------------------------------------------------------------------------
# send_reply — mocked HTTP
# ---------------------------------------------------------------------------


class TestTelegramBotPollerReply:
    @pytest.fixture()
    def poller(self) -> TelegramBotPoller:
        return TelegramBotPoller(
            bot_token="tok",
            chat_id="999",
            timeout=1,
            message_char_limit=50,
        )

    async def test_send_reply_success(self, poller: TelegramBotPoller) -> None:
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)

        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        with patch(target, return_value=resp) as mock:
            ok = await poller.send_reply("hello")

        assert ok is True
        req = mock.call_args[0][0]
        assert "/bottok/sendMessage" in req.full_url
        body = json.loads(req.data)
        assert body == {"chat_id": "999", "text": "hello"}

    async def test_send_reply_truncates_long_text(self, poller: TelegramBotPoller) -> None:
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)

        long_text = "x" * 200
        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        with patch(target, return_value=resp) as mock:
            await poller.send_reply(long_text)

        body = json.loads(mock.call_args[0][0].data)
        assert len(body["text"]) == 50  # message_char_limit
        assert body["text"].endswith("...")

    async def test_send_reply_empty_text_returns_false(self, poller: TelegramBotPoller) -> None:
        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        with patch(target) as mock:
            ok = await poller.send_reply("")
        assert ok is False
        mock.assert_not_called()

    async def test_send_reply_http_error_returns_false(self, poller: TelegramBotPoller) -> None:
        exc = urllib.error.HTTPError(
            url="https://api.telegram.org",
            code=403,
            msg="Forbidden",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b""),
        )
        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        with patch(target, side_effect=exc):
            ok = await poller.send_reply("hello")
        assert ok is False

    async def test_send_reply_url_error_returns_false(self, poller: TelegramBotPoller) -> None:
        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        with patch(target, side_effect=urllib.error.URLError("offline")):
            ok = await poller.send_reply("hello")
        assert ok is False


# ---------------------------------------------------------------------------
# allowed_chat_ids — multi-chat auth
# ---------------------------------------------------------------------------


class TestAllowedChatIds:
    def test_defaults_to_primary_only(self) -> None:
        poller = TelegramBotPoller(bot_token="tok", chat_id="999")
        assert poller.allowed_chat_ids == ("999",)

    def test_primary_always_authorised(self) -> None:
        poller = TelegramBotPoller(bot_token="tok", chat_id="999", allowed_chat_ids=["1"])
        assert "999" in poller.allowed_chat_ids
        assert "1" in poller.allowed_chat_ids

    def test_duplicates_deduped(self) -> None:
        poller = TelegramBotPoller(
            bot_token="tok", chat_id="999", allowed_chat_ids=["999", "1", "1"]
        )
        assert poller.allowed_chat_ids == ("999", "1")

    def test_blank_entries_dropped(self) -> None:
        poller = TelegramBotPoller(bot_token="tok", chat_id="999", allowed_chat_ids=["", "  ", "1"])
        assert poller.allowed_chat_ids == ("999", "1")

    async def test_poll_accepts_secondary_chat(self) -> None:
        poller = TelegramBotPoller(
            bot_token="tok",
            chat_id="999",
            allowed_chat_ids=["42"],
            timeout=1,
        )
        payload = {
            "ok": True,
            "result": [
                _make_update(1, 999, "/status"),
                _make_update(2, 42, "/alerts"),
                _make_update(3, 777, "/leak"),  # unauthorised
            ],
        }
        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        with patch(target, return_value=_mock_response(payload)):
            commands = await poller.poll()

        actions = [c.action for c in commands]
        assert actions == ["status", "alerts"]
        chats = [c.chat_id for c in commands]
        assert chats == ["999", "42"]

    async def test_send_reply_routes_to_secondary_chat(self) -> None:
        poller = TelegramBotPoller(
            bot_token="tok",
            chat_id="999",
            allowed_chat_ids=["42"],
            timeout=1,
        )
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        with patch(target, return_value=resp) as mock:
            ok = await poller.send_reply("hi", chat_id="42")
        assert ok is True
        body = json.loads(mock.call_args[0][0].data)
        assert body["chat_id"] == "42"

    async def test_send_reply_unauthorised_chat_falls_back_to_primary(self) -> None:
        poller = TelegramBotPoller(
            bot_token="tok",
            chat_id="999",
            allowed_chat_ids=["42"],
            timeout=1,
        )
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        with patch(target, return_value=resp) as mock:
            ok = await poller.send_reply("hi", chat_id="8888")
        assert ok is True
        body = json.loads(mock.call_args[0][0].data)
        assert body["chat_id"] == "999"

    async def test_send_reply_default_uses_primary(self) -> None:
        poller = TelegramBotPoller(
            bot_token="tok",
            chat_id="999",
            allowed_chat_ids=["42"],
            timeout=1,
        )
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        with patch(target, return_value=resp) as mock:
            await poller.send_reply("hi")
        body = json.loads(mock.call_args[0][0].data)
        assert body["chat_id"] == "999"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimit:
    def test_rate_limit_commands_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="rate_limit_commands"):
            TelegramBotPoller(bot_token="tok", chat_id="999", rate_limit_commands=-1)

    def test_rate_limit_window_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="rate_limit_window_seconds"):
            TelegramBotPoller(bot_token="tok", chat_id="999", rate_limit_window_seconds=0)

    async def test_poll_drops_commands_over_limit(self) -> None:
        poller = TelegramBotPoller(
            bot_token="tok",
            chat_id="999",
            timeout=1,
            rate_limit_commands=2,
            rate_limit_window_seconds=60.0,
        )
        payload = {
            "ok": True,
            "result": [
                _make_update(1, 999, "/status"),
                _make_update(2, 999, "/alerts"),
                _make_update(3, 999, "/tasks"),  # dropped
                _make_update(4, 999, "/status"),  # dropped
            ],
        }
        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        with patch(target, return_value=_mock_response(payload)):
            commands = await poller.poll()

        actions = [c.action for c in commands]
        assert actions == ["status", "alerts"]
        # Offset still advances past the dropped updates so we don't
        # re-fetch them on the next poll.
        assert poller.offset == 5

    async def test_rate_limit_is_per_chat(self) -> None:
        poller = TelegramBotPoller(
            bot_token="tok",
            chat_id="999",
            allowed_chat_ids=["42"],
            timeout=1,
            rate_limit_commands=1,
            rate_limit_window_seconds=60.0,
        )
        payload = {
            "ok": True,
            "result": [
                _make_update(1, 999, "/status"),
                _make_update(2, 42, "/status"),  # different chat, own bucket
                _make_update(3, 999, "/alerts"),  # dropped
                _make_update(4, 42, "/alerts"),  # dropped
            ],
        }
        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        with patch(target, return_value=_mock_response(payload)):
            commands = await poller.poll()

        chats = [(c.chat_id, c.action) for c in commands]
        assert chats == [("999", "status"), ("42", "status")]

    async def test_rate_limit_window_expires(self) -> None:
        import qracer.notifications.telegram_poller as mod

        poller = TelegramBotPoller(
            bot_token="tok",
            chat_id="999",
            timeout=1,
            rate_limit_commands=1,
            rate_limit_window_seconds=60.0,
        )
        payload_first = {
            "ok": True,
            "result": [_make_update(1, 999, "/status")],
        }
        payload_second = {
            "ok": True,
            "result": [_make_update(2, 999, "/alerts")],
        }
        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"

        # First poll at t=0 — admitted.
        with (
            patch.object(mod.time, "monotonic", return_value=0.0),
            patch(target, return_value=_mock_response(payload_first)),
        ):
            first = await poller.poll()
        assert len(first) == 1

        # Second poll at t=120s (> window) — the earlier timestamp has aged
        # out and the new command is admitted again.
        with (
            patch.object(mod.time, "monotonic", return_value=120.0),
            patch(target, return_value=_mock_response(payload_second)),
        ):
            second = await poller.poll()
        assert len(second) == 1
        assert second[0].action == "alerts"

    async def test_rate_limit_zero_blocks_everything(self) -> None:
        poller = TelegramBotPoller(
            bot_token="tok",
            chat_id="999",
            timeout=1,
            rate_limit_commands=0,
            rate_limit_window_seconds=60.0,
        )
        payload = {
            "ok": True,
            "result": [_make_update(1, 999, "/status")],
        }
        target = "qracer.notifications.telegram_poller.urllib.request.urlopen"
        with patch(target, return_value=_mock_response(payload)):
            commands = await poller.poll()
        assert commands == []


# ---------------------------------------------------------------------------
# BotCommand.chat_id plumbing
# ---------------------------------------------------------------------------


class TestBotCommandChatId:
    def test_parse_default_blank(self) -> None:
        cmd = BotCommand.parse("/status")
        assert cmd is not None
        assert cmd.chat_id == ""

    def test_parse_records_chat_id(self) -> None:
        cmd = BotCommand.parse("/status", chat_id="12345")
        assert cmd is not None
        assert cmd.chat_id == "12345"

    def test_parse_coerces_chat_id_to_str(self) -> None:
        cmd = BotCommand.parse("/status", chat_id=12345)  # type: ignore[arg-type]
        assert cmd is not None
        assert cmd.chat_id == "12345"
