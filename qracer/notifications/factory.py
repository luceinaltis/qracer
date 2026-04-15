"""Factory for building a NotificationRegistry from QracerConfig credentials.

Usage::

    from qracer.config.loader import load_config
    from qracer.notifications.factory import build_notification_registry

    config = load_config()
    registry = build_notification_registry(config.credentials)
"""

from __future__ import annotations

import logging

from qracer.notifications.registry import NotificationRegistry
from qracer.notifications.telegram_adapter import TelegramAdapter
from qracer.notifications.telegram_poller import TelegramBotPoller

logger = logging.getLogger(__name__)


def build_notification_registry(
    credentials: dict[str, str],
) -> NotificationRegistry:
    """Create a :class:`NotificationRegistry` with channels enabled by credentials.

    Currently supports:
    - **Telegram**: requires ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID``.
    """
    registry = NotificationRegistry()

    bot_token = credentials.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = credentials.get("TELEGRAM_CHAT_ID", "")

    if bot_token and chat_id:
        adapter = TelegramAdapter(bot_token=bot_token, chat_id=chat_id)
        registry.register("telegram", adapter)
        logger.info("Telegram notification channel registered")
    else:
        logger.debug(
            "Telegram credentials not found — skipping Telegram channel "
            "(set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in credentials.env)"
        )

    return registry


def _parse_chat_ids(raw: str) -> list[str]:
    """Split a comma-separated chat-id list and drop blanks."""
    return [part.strip() for part in raw.split(",") if part.strip()]


def build_telegram_poller(
    credentials: dict[str, str],
    *,
    timeout: int = 1,
) -> TelegramBotPoller | None:
    """Build a :class:`TelegramBotPoller` if Telegram credentials are present.

    Returns ``None`` when ``TELEGRAM_BOT_TOKEN`` or ``TELEGRAM_CHAT_ID`` are
    missing — callers should treat this as "no inbound bot integration".

    The default ``timeout=1`` keeps the long-poll short enough to coexist
    with the 1-second :class:`~qracer.server.Server` tick; standalone
    callers can pass a larger value (e.g. 30) for true long-polling.

    ``TELEGRAM_ALLOWED_CHAT_IDS`` (comma-separated, optional) authorises
    additional chats — e.g. ``"111,222"`` lets two users talk to the bot.
    The primary chat (``TELEGRAM_CHAT_ID``) is always authorised and used as
    the default reply target.
    """
    bot_token = credentials.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = credentials.get("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        return None
    allowed = _parse_chat_ids(credentials.get("TELEGRAM_ALLOWED_CHAT_IDS", ""))
    poller = TelegramBotPoller(
        bot_token=bot_token,
        chat_id=chat_id,
        allowed_chat_ids=allowed or None,
        timeout=timeout,
    )
    if len(poller.allowed_chat_ids) > 1:
        logger.info(
            "Telegram bot command poller initialised (authorised chats: %d)",
            len(poller.allowed_chat_ids),
        )
    else:
        logger.info("Telegram bot command poller initialised")
    return poller
