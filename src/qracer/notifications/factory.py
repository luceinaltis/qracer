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
