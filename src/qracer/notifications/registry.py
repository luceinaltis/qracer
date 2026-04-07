"""NotificationRegistry — channel-based routing with fan-out.

Adapters register under a channel name.  ``notify()`` fans out a
notification to every registered channel and returns the per-channel
success map.
"""

from __future__ import annotations

import logging
from typing import Any

from qracer.notifications.providers import Notification, NotificationProvider

logger = logging.getLogger(__name__)


class NotificationRegistry:
    """Routes notifications to one or more channel adapters."""

    def __init__(self) -> None:
        self._channels: dict[str, NotificationProvider] = {}

    def register(self, name: str, adapter: Any) -> None:
        """Register a notification channel adapter.

        Args:
            name: Unique channel name (e.g. ``"telegram"``).
            adapter: An object implementing :class:`NotificationProvider`.
        """
        if not isinstance(adapter, NotificationProvider):
            raise TypeError(
                f"Adapter '{name}' does not implement NotificationProvider protocol"
            )
        self._channels[name] = adapter

    def unregister(self, name: str) -> None:
        """Remove a channel by name (no-op if not registered)."""
        self._channels.pop(name, None)

    @property
    def channels(self) -> list[str]:
        """Return the names of all registered channels."""
        return list(self._channels)

    async def notify(self, notification: Notification) -> dict[str, bool]:
        """Fan-out *notification* to every registered channel.

        Returns a mapping of channel name → delivery success.
        """
        results: dict[str, bool] = {}
        for name, adapter in self._channels.items():
            try:
                ok = await adapter.send(notification)
                results[name] = ok
            except Exception:
                logger.exception("Channel '%s' raised during send", name)
                results[name] = False
        return results
