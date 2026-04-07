"""Tests for NotificationRegistry."""

import pytest

from qracer.notifications.providers import Notification, NotificationCategory
from qracer.notifications.registry import NotificationRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _notif(title: str = "test") -> Notification:
    return Notification(
        category=NotificationCategory.PRICE_ALERT,
        title=title,
        body="body",
    )


class FakeChannel:
    """Fake notification channel that records calls."""

    def __init__(self, succeed: bool = True) -> None:
        self.sent: list[Notification] = []
        self._succeed = succeed

    async def send(self, notification: Notification) -> bool:
        self.sent.append(notification)
        return self._succeed


class ExplodingChannel:
    """Channel that raises on send."""

    async def send(self, notification: Notification) -> bool:
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNotificationRegistry:
    def test_register_and_list_channels(self):
        reg = NotificationRegistry()
        reg.register("telegram", FakeChannel())
        reg.register("email", FakeChannel())
        assert sorted(reg.channels) == ["email", "telegram"]

    def test_register_rejects_non_provider(self):
        reg = NotificationRegistry()
        with pytest.raises(TypeError, match="NotificationProvider"):
            reg.register("bad", object())  # type: ignore[arg-type]

    def test_unregister(self):
        reg = NotificationRegistry()
        reg.register("telegram", FakeChannel())
        reg.unregister("telegram")
        assert reg.channels == []

    def test_unregister_missing_is_noop(self):
        reg = NotificationRegistry()
        reg.unregister("nonexistent")  # should not raise

    async def test_notify_fanout(self):
        ch1 = FakeChannel()
        ch2 = FakeChannel()
        reg = NotificationRegistry()
        reg.register("a", ch1)
        reg.register("b", ch2)

        n = _notif("hello")
        results = await reg.notify(n)

        assert results == {"a": True, "b": True}
        assert ch1.sent == [n]
        assert ch2.sent == [n]

    async def test_notify_partial_failure(self):
        ok = FakeChannel(succeed=True)
        fail = FakeChannel(succeed=False)
        reg = NotificationRegistry()
        reg.register("ok", ok)
        reg.register("fail", fail)

        results = await reg.notify(_notif())
        assert results["ok"] is True
        assert results["fail"] is False

    async def test_notify_exception_is_caught(self):
        reg = NotificationRegistry()
        reg.register("boom", ExplodingChannel())
        reg.register("ok", FakeChannel())

        results = await reg.notify(_notif())
        assert results["boom"] is False
        assert results["ok"] is True

    async def test_notify_empty_registry(self):
        reg = NotificationRegistry()
        results = await reg.notify(_notif())
        assert results == {}
