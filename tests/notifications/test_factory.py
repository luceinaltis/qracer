"""Tests for the notification factory."""

from qracer.notifications.factory import build_notification_registry


class TestBuildNotificationRegistry:
    def test_telegram_registered_when_credentials_present(self):
        creds = {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_CHAT_ID": "123"}
        reg = build_notification_registry(creds)
        assert "telegram" in reg.channels

    def test_telegram_skipped_when_token_missing(self):
        creds = {"TELEGRAM_CHAT_ID": "123"}
        reg = build_notification_registry(creds)
        assert reg.channels == []

    def test_telegram_skipped_when_chat_id_missing(self):
        creds = {"TELEGRAM_BOT_TOKEN": "tok"}
        reg = build_notification_registry(creds)
        assert reg.channels == []

    def test_empty_credentials(self):
        reg = build_notification_registry({})
        assert reg.channels == []

    def test_empty_string_values_are_skipped(self):
        creds = {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}
        reg = build_notification_registry(creds)
        assert reg.channels == []
