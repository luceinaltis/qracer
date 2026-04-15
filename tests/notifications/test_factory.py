"""Tests for the notification factory."""

from qracer.notifications.factory import (
    build_notification_registry,
    build_telegram_poller,
)


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


class TestBuildTelegramPoller:
    def test_returns_none_without_credentials(self):
        assert build_telegram_poller({}) is None

    def test_returns_none_without_bot_token(self):
        assert build_telegram_poller({"TELEGRAM_CHAT_ID": "1"}) is None

    def test_returns_none_without_chat_id(self):
        assert build_telegram_poller({"TELEGRAM_BOT_TOKEN": "tok"}) is None

    def test_primary_chat_authorised_by_default(self):
        poller = build_telegram_poller({"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_CHAT_ID": "1"})
        assert poller is not None
        assert poller.allowed_chat_ids == ("1",)

    def test_allowed_chat_ids_parsed_from_comma_list(self):
        poller = build_telegram_poller(
            {
                "TELEGRAM_BOT_TOKEN": "tok",
                "TELEGRAM_CHAT_ID": "1",
                "TELEGRAM_ALLOWED_CHAT_IDS": "2, 3 ,4",
            }
        )
        assert poller is not None
        # Whitespace trimmed, primary still present, duplicates deduped.
        assert poller.allowed_chat_ids == ("1", "2", "3", "4")

    def test_allowed_chat_ids_blanks_ignored(self):
        poller = build_telegram_poller(
            {
                "TELEGRAM_BOT_TOKEN": "tok",
                "TELEGRAM_CHAT_ID": "1",
                "TELEGRAM_ALLOWED_CHAT_IDS": ",,  ,",
            }
        )
        assert poller is not None
        assert poller.allowed_chat_ids == ("1",)

    def test_allowed_chat_ids_dedupes_primary(self):
        poller = build_telegram_poller(
            {
                "TELEGRAM_BOT_TOKEN": "tok",
                "TELEGRAM_CHAT_ID": "1",
                "TELEGRAM_ALLOWED_CHAT_IDS": "1,2",
            }
        )
        assert poller is not None
        assert poller.allowed_chat_ids == ("1", "2")
