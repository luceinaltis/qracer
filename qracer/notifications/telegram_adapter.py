"""TelegramAdapter — deliver notifications via Telegram Bot API.

Requires two credentials (from ``credentials.env``):
- ``TELEGRAM_BOT_TOKEN``  — the bot token from @BotFather
- ``TELEGRAM_CHAT_ID``    — the target chat/group/channel ID

Uses only the stdlib ``urllib`` so there is no extra dependency.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from qracer.notifications.providers import Notification

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org"


def _format_message(notification: Notification) -> str:
    """Build a human-readable Telegram message from a Notification."""
    category_label = notification.category.value.replace("_", " ").title()
    return f"*[{category_label}]*\n\n*{notification.title}*\n{notification.body}"


class TelegramAdapter:
    """Notification adapter for Telegram via the Bot HTTP API."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        if not bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required but was empty")
        if not chat_id:
            raise ValueError("TELEGRAM_CHAT_ID is required but was empty")
        self._bot_token = bot_token
        self._chat_id = chat_id

    async def send(self, notification: Notification) -> bool:
        """Send *notification* as a Telegram message.

        Uses ``urllib`` (sync, but fast enough for a single POST) wrapped in
        the async interface so callers can treat all channels uniformly.
        """
        text = _format_message(notification)
        url = f"{_TELEGRAM_API}/bot{self._bot_token}/sendMessage"
        payload: dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }
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
                    logger.warning("Telegram API returned status %s", resp.status)
                return ok
        except urllib.error.HTTPError as exc:
            logger.error("Telegram send failed (HTTP %s): %s", exc.code, exc.reason)
            return False
        except urllib.error.URLError as exc:
            logger.error("Telegram send failed (network): %s", exc.reason)
            return False
