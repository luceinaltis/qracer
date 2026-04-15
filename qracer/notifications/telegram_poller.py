"""TelegramBotPoller — receive bot commands from Telegram via long-polling.

Companion to :mod:`qracer.notifications.telegram_adapter` (which only sends).
This module adds inbound command receiving so users can interact with the
qracer service remotely from their phone.

Uses only the stdlib ``urllib`` so there is no extra dependency, mirroring
the existing TelegramAdapter design.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org"

# Telegram caps a single message at 4096 characters; leave a small margin so
# truncation suffixes still fit.
_DEFAULT_MESSAGE_CHAR_LIMIT = 4000

# Default rate-limit: 20 commands per chat per 60 seconds. Balances
# responsiveness for normal use against runaway loops or abuse on a shared
# chat.
_DEFAULT_RATE_LIMIT_COMMANDS = 20
_DEFAULT_RATE_LIMIT_WINDOW_S = 60.0


@dataclass(frozen=True)
class BotCommand:
    """A parsed bot command from a Telegram message.

    Example::

        BotCommand.parse("/analyze AAPL", chat_id="12345")
        # → BotCommand(action="analyze", args=["AAPL"],
        #              raw_text="/analyze AAPL", chat_id="12345")

    ``chat_id`` is the sender's chat — callers can use it as the target of
    :meth:`TelegramBotPoller.send_reply` so replies go back to whoever asked
    (useful when ``allowed_chat_ids`` authorises more than one chat).
    """

    action: str
    args: list[str]
    raw_text: str
    chat_id: str = ""

    @classmethod
    def parse(cls, text: str, chat_id: str = "") -> BotCommand | None:
        """Parse a Telegram message into a :class:`BotCommand`.

        Returns ``None`` if the text is not a recognised command (i.e. does
        not start with ``/`` or contains no action token).

        Telegram bot commands may include a ``@botname`` suffix when sent in
        groups (e.g. ``/analyze@qracerbot AAPL``); the suffix is stripped so
        the action is just ``analyze``.
        """
        text = (text or "").strip()
        if not text or not text.startswith("/"):
            return None
        parts = text[1:].split()
        if not parts:
            return None
        action = parts[0].split("@", 1)[0].lower()
        if not action:
            return None
        return cls(action=action, args=parts[1:], raw_text=text, chat_id=str(chat_id))


@dataclass
class _RateBucket:
    """Sliding-window counter for a single chat."""

    timestamps: deque[float] = field(default_factory=deque)

    def admit(self, now: float, limit: int, window: float) -> bool:
        """Return ``True`` if a new command is within limits at ``now``."""
        cutoff = now - window
        while self.timestamps and self.timestamps[0] <= cutoff:
            self.timestamps.popleft()
        if len(self.timestamps) >= limit:
            return False
        self.timestamps.append(now)
        return True


class TelegramBotPoller:
    """Receive bot commands from Telegram via the ``getUpdates`` long-poll API.

    Tracks the update offset so messages are never returned twice, filters
    messages to those originating from an authorised chat (``chat_id`` plus
    any ``allowed_chat_ids``), parses incoming text into :class:`BotCommand`
    objects, and enforces a per-chat sliding-window rate limit.

    Replies can be sent back to any authorised chat via :meth:`send_reply`;
    when ``chat_id`` is omitted the primary chat (``self.chat_id``) is used.

    Usage::

        poller = TelegramBotPoller(
            bot_token="...",
            chat_id="123",
            allowed_chat_ids=["123", "456"],
        )
        commands = await poller.poll()
        for cmd in commands:
            await poller.send_reply(f"Got: {cmd.action}", chat_id=cmd.chat_id)
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        *,
        allowed_chat_ids: list[str] | None = None,
        timeout: int = 30,
        message_char_limit: int = _DEFAULT_MESSAGE_CHAR_LIMIT,
        rate_limit_commands: int = _DEFAULT_RATE_LIMIT_COMMANDS,
        rate_limit_window_seconds: float = _DEFAULT_RATE_LIMIT_WINDOW_S,
    ) -> None:
        if not bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required but was empty")
        if not chat_id:
            raise ValueError("TELEGRAM_CHAT_ID is required but was empty")
        self._bot_token = bot_token
        self._chat_id = str(chat_id)

        # Authorised senders. Always include the primary chat_id; merge any
        # extras while preserving insertion order and dropping blanks.
        authorised: list[str] = [self._chat_id]
        for extra in allowed_chat_ids or []:
            extra_str = str(extra).strip()
            if extra_str and extra_str not in authorised:
                authorised.append(extra_str)
        self._allowed_chat_ids: tuple[str, ...] = tuple(authorised)

        self._timeout = max(0, int(timeout))
        self._message_char_limit = message_char_limit
        self._offset: int | None = None

        if rate_limit_commands < 0:
            raise ValueError("rate_limit_commands must be >= 0")
        if rate_limit_window_seconds <= 0:
            raise ValueError("rate_limit_window_seconds must be > 0")
        self._rate_limit_commands = rate_limit_commands
        self._rate_limit_window = rate_limit_window_seconds
        self._rate_buckets: dict[str, _RateBucket] = {}

    @property
    def offset(self) -> int | None:
        """Current update offset (``None`` until the first update arrives)."""
        return self._offset

    @property
    def chat_id(self) -> str:
        """The primary chat ID — default target for :meth:`send_reply`."""
        return self._chat_id

    @property
    def allowed_chat_ids(self) -> tuple[str, ...]:
        """All chat IDs authorised to send commands (primary first)."""
        return self._allowed_chat_ids

    async def poll(self) -> list[BotCommand]:
        """Long-poll Telegram for new commands.

        Returns a list of :class:`BotCommand` parsed from messages that
        arrived from an authorised chat. The offset is advanced past the
        highest update ID returned, so subsequent calls only return new
        messages. Commands that exceed the per-chat rate limit are logged
        and dropped.

        Network and API errors are logged and converted to an empty list
        — the caller is expected to retry on the next tick.
        """
        return await asyncio.to_thread(self._poll_sync)

    def _poll_sync(self) -> list[BotCommand]:
        url = f"{_TELEGRAM_API}/bot{self._bot_token}/getUpdates"
        params: dict[str, Any] = {
            "timeout": self._timeout,
            "allowed_updates": json.dumps(["message"]),
        }
        if self._offset is not None:
            params["offset"] = self._offset
        full_url = f"{url}?{urllib.parse.urlencode(params)}"

        try:
            with urllib.request.urlopen(full_url, timeout=self._timeout + 5) as resp:
                if resp.status != 200:
                    logger.warning("Telegram getUpdates returned status %s", resp.status)
                    return []
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            logger.error("Telegram getUpdates failed (HTTP %s): %s", exc.code, exc.reason)
            return []
        except urllib.error.URLError as exc:
            logger.error("Telegram getUpdates failed (network): %s", exc.reason)
            return []
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Telegram getUpdates returned invalid JSON: %s", exc)
            return []

        if not isinstance(payload, dict) or not payload.get("ok"):
            logger.warning(
                "Telegram getUpdates returned ok=false: %s",
                payload.get("description") if isinstance(payload, dict) else payload,
            )
            return []

        commands: list[BotCommand] = []
        max_update_id = -1
        now = time.monotonic()
        for update in payload.get("result", []):
            update_id = update.get("update_id")
            if isinstance(update_id, int) and update_id > max_update_id:
                max_update_id = update_id

            message = update.get("message")
            if not isinstance(message, dict):
                continue

            chat = message.get("chat") or {}
            sender_chat_id = str(chat.get("id"))
            if sender_chat_id not in self._allowed_chat_ids:
                logger.debug(
                    "Ignoring message from unauthorised chat %s",
                    sender_chat_id,
                )
                continue

            text = message.get("text")
            if not isinstance(text, str):
                continue

            cmd = BotCommand.parse(text, chat_id=sender_chat_id)
            if cmd is None:
                continue

            if not self._admit(sender_chat_id, now):
                logger.warning(
                    "Rate-limited command from chat %s: /%s",
                    sender_chat_id,
                    cmd.action,
                )
                continue
            commands.append(cmd)

        if max_update_id >= 0:
            self._offset = max_update_id + 1

        return commands

    def _admit(self, chat_id: str, now: float) -> bool:
        """Return True when this chat is within the sliding-window limit."""
        if self._rate_limit_commands == 0:
            return False
        bucket = self._rate_buckets.get(chat_id)
        if bucket is None:
            bucket = _RateBucket()
            self._rate_buckets[chat_id] = bucket
        return bucket.admit(now, self._rate_limit_commands, self._rate_limit_window)

    async def send_reply(self, text: str, chat_id: str | None = None) -> bool:
        """Send a plain-text reply.

        ``chat_id`` defaults to the primary :attr:`chat_id`; pass an explicit
        value to reply to a secondary authorised chat (e.g. the sender's
        :attr:`BotCommand.chat_id`). Unknown chat IDs fall back to the
        primary chat with a warning log.

        Long replies are truncated to ``message_char_limit`` characters with
        a trailing ``"..."``. Returns ``True`` on HTTP 200.
        """
        target = chat_id if chat_id else self._chat_id
        if target not in self._allowed_chat_ids:
            logger.warning(
                "send_reply called with unauthorised chat %s; falling back to primary",
                target,
            )
            target = self._chat_id
        return await asyncio.to_thread(self._send_reply_sync, text, target)

    def _send_reply_sync(self, text: str, chat_id: str) -> bool:
        if not text:
            return False
        if len(text) > self._message_char_limit:
            text = text[: self._message_char_limit - 3] + "..."

        url = f"{_TELEGRAM_API}/bot{self._bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text}
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                ok = resp.status == 200
                if not ok:
                    logger.warning("Telegram sendMessage returned status %s", resp.status)
                return ok
        except urllib.error.HTTPError as exc:
            logger.error("Telegram sendMessage failed (HTTP %s): %s", exc.code, exc.reason)
            return False
        except urllib.error.URLError as exc:
            logger.error("Telegram sendMessage failed (network): %s", exc.reason)
            return False
