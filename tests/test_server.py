"""Tests for the Server loop."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from qracer.alerts import Alert, AlertCondition
from qracer.notifications.telegram_poller import BotCommand
from qracer.server import Server, _format_duration
from qracer.tasks import Task, TaskActionType, TaskScheduleType, TaskStatus


def _make_monitor(triggered=None):
    monitor = MagicMock()
    monitor.should_check.return_value = True
    monitor.check = AsyncMock(return_value=triggered or [])
    return monitor


def _make_executor(results=None):
    executor = MagicMock()
    executor.should_check.return_value = True
    executor.check = AsyncMock(return_value=results or [])
    return executor


def _make_poller():
    poller = MagicMock()
    poller.poll = AsyncMock(return_value=[])
    poller.send_reply = AsyncMock(return_value=True)
    return poller


class TestServer:
    async def test_tick_checks_alerts_and_tasks(self) -> None:
        monitor = _make_monitor()
        executor = _make_executor()
        server = Server(monitor, executor)

        await server._tick()

        monitor.should_check.assert_called_once()
        monitor.check.assert_called_once()
        executor.should_check.assert_called_once()
        executor.check.assert_called_once()

    async def test_tick_skips_when_not_due(self) -> None:
        monitor = _make_monitor()
        monitor.should_check.return_value = False
        executor = _make_executor()
        executor.should_check.return_value = False
        server = Server(monitor, executor)

        await server._tick()

        monitor.check.assert_not_called()
        executor.check.assert_not_called()

    async def test_tick_sends_alert_notification(self) -> None:
        alert_result = MagicMock()
        alert_result.message = "AAPL above 200"
        monitor = _make_monitor(triggered=[alert_result])
        executor = _make_executor()
        notifications = MagicMock()
        notifications.channels = ["telegram"]
        notifications.notify = AsyncMock(return_value={"telegram": True})

        server = Server(monitor, executor, notifications)
        await server._tick()

        notifications.notify.assert_called_once()
        notification = notifications.notify.call_args[0][0]
        assert "AAPL" in notification.title

    async def test_tick_sends_task_failure_notification(self) -> None:
        task_result = MagicMock()
        task_result.success = False
        task_result.task.describe.return_value = "analyze AAPL"
        task_result.error = "connection error"
        monitor = _make_monitor()
        executor = _make_executor(results=[task_result])
        notifications = MagicMock()
        notifications.channels = ["telegram"]
        notifications.notify = AsyncMock(return_value={"telegram": True})

        server = Server(monitor, executor, notifications)
        await server._tick()

        notifications.notify.assert_called_once()

    async def test_tick_no_notification_without_channels(self) -> None:
        alert_result = MagicMock()
        alert_result.message = "AAPL above 200"
        monitor = _make_monitor(triggered=[alert_result])
        executor = _make_executor()
        notifications = MagicMock()
        notifications.channels = []  # No channels

        server = Server(monitor, executor, notifications)
        await server._tick()

        notifications.notify.assert_not_called()

    async def test_shutdown_stops_run(self) -> None:
        monitor = _make_monitor()
        executor = _make_executor()
        server = Server(monitor, executor, tick_interval=0.05)

        async def stop_after_delay():
            await asyncio.sleep(0.1)
            server.shutdown()

        await asyncio.gather(server.run(), stop_after_delay())
        # If we get here, shutdown worked

    async def test_tick_handles_alert_error_gracefully(self) -> None:
        monitor = _make_monitor()
        monitor.check = AsyncMock(side_effect=RuntimeError("boom"))
        executor = _make_executor()
        server = Server(monitor, executor)

        await server._tick()  # Should not raise

        executor.check.assert_called_once()  # Tasks still checked

    async def test_tick_handles_task_error_gracefully(self) -> None:
        monitor = _make_monitor()
        executor = _make_executor()
        executor.check = AsyncMock(side_effect=RuntimeError("boom"))
        server = Server(monitor, executor)

        await server._tick()  # Should not raise


# ---------------------------------------------------------------------------
# Telegram bot command integration
# ---------------------------------------------------------------------------


def _alert(id_: str, ticker: str, threshold: float, active: bool = True) -> Alert:
    return Alert(
        id=id_,
        ticker=ticker,
        condition=AlertCondition.ABOVE,
        threshold=threshold,
        created_at="2026-01-01T00:00:00+00:00",
        active=active,
    )


def _task(id_: str, ticker: str, schedule: str = "every 1h") -> Task:
    return Task(
        id=id_,
        action_type=TaskActionType.ANALYZE,
        action_params={"ticker": ticker},
        schedule_type=TaskScheduleType.RECURRING,
        schedule_spec=schedule,
        status=TaskStatus.PENDING,
        created_at="2026-01-01T00:00:00+00:00",
        next_run_at="2026-01-01T01:00:00+00:00",
    )


class TestServerTelegramPoller:
    async def test_tick_polls_telegram_when_configured(self) -> None:
        monitor = _make_monitor()
        executor = _make_executor()
        poller = _make_poller()
        server = Server(monitor, executor, telegram_poller=poller)

        await server._tick()

        poller.poll.assert_awaited_once()

    async def test_tick_skips_telegram_when_not_configured(self) -> None:
        monitor = _make_monitor()
        executor = _make_executor()
        server = Server(monitor, executor)

        await server._tick()  # No poller — should not raise.

    async def test_tick_dispatches_command_and_replies(self) -> None:
        monitor = _make_monitor()
        monitor.store.get_active.return_value = []
        executor = _make_executor()
        poller = _make_poller()
        poller.poll = AsyncMock(
            return_value=[BotCommand(action="alerts", args=[], raw_text="/alerts")]
        )
        server = Server(monitor, executor, telegram_poller=poller)

        await server._tick()

        poller.send_reply.assert_awaited_once()
        assert "No active alerts" in poller.send_reply.await_args[0][0]

    async def test_tick_handles_poll_failure_gracefully(self) -> None:
        monitor = _make_monitor()
        executor = _make_executor()
        poller = _make_poller()
        poller.poll = AsyncMock(side_effect=RuntimeError("boom"))
        server = Server(monitor, executor, telegram_poller=poller)

        await server._tick()  # Should not raise

        poller.send_reply.assert_not_called()

    async def test_tick_handles_dispatch_exception_with_error_reply(self) -> None:
        monitor = _make_monitor()
        monitor.store.get_active.side_effect = RuntimeError("db down")
        executor = _make_executor()
        poller = _make_poller()
        poller.poll = AsyncMock(
            return_value=[BotCommand(action="alerts", args=[], raw_text="/alerts")]
        )
        server = Server(monitor, executor, telegram_poller=poller)

        await server._tick()

        poller.send_reply.assert_awaited_once()
        msg = poller.send_reply.await_args[0][0]
        assert "Error handling /alerts" in msg
        assert "db down" in msg


class TestBotCommandHandlers:
    @staticmethod
    def _server(monitor=None, executor=None, **kwargs) -> Server:
        return Server(
            monitor or _make_monitor(),
            executor or _make_executor(),
            **kwargs,
        )

    async def test_help(self) -> None:
        server = self._server()
        out = await server._dispatch_bot_command(BotCommand("help", [], "/help"))
        assert "/status" in out
        assert "/alerts" in out
        assert "/alert" in out
        assert "/tasks" in out
        assert "/schedule" in out

    async def test_start_aliases_help(self) -> None:
        server = self._server()
        out = await server._dispatch_bot_command(BotCommand("start", [], "/start"))
        assert "/status" in out

    async def test_unknown_command(self) -> None:
        server = self._server()
        out = await server._dispatch_bot_command(BotCommand("nope", [], "/nope"))
        assert "Unknown command" in out

    async def test_analyze_returns_not_supported(self) -> None:
        server = self._server()
        out = await server._dispatch_bot_command(BotCommand("analyze", ["AAPL"], "/analyze AAPL"))
        assert "not supported" in out.lower()

    async def test_portfolio_returns_not_supported(self) -> None:
        server = self._server()
        out = await server._dispatch_bot_command(BotCommand("portfolio", [], "/portfolio"))
        assert "not supported" in out.lower()

    async def test_status(self) -> None:
        notifications = MagicMock()
        notifications.channels = ["telegram"]
        server = self._server(notifications=notifications)
        out = await server._dispatch_bot_command(BotCommand("status", [], "/status"))
        assert "uptime" in out
        assert "telegram" in out
        assert "autonomous: off" in out

    async def test_alerts_empty(self) -> None:
        monitor = _make_monitor()
        monitor.store.get_active.return_value = []
        server = self._server(monitor=monitor)
        out = await server._dispatch_bot_command(BotCommand("alerts", [], "/alerts"))
        assert out == "No active alerts."

    async def test_alerts_lists_each(self) -> None:
        monitor = _make_monitor()
        monitor.store.get_active.return_value = [
            _alert("a1", "AAPL", 200),
            _alert("b2", "MSFT", 410),
        ]
        server = self._server(monitor=monitor)
        out = await server._dispatch_bot_command(BotCommand("alerts", [], "/alerts"))
        assert "a1" in out
        assert "AAPL" in out
        assert "b2" in out
        assert "MSFT" in out

    async def test_create_alert_validates_args(self) -> None:
        server = self._server()
        out = await server._dispatch_bot_command(BotCommand("alert", ["AAPL"], "/alert AAPL"))
        assert "Usage" in out

    async def test_create_alert_rejects_unknown_condition(self) -> None:
        server = self._server()
        out = await server._dispatch_bot_command(
            BotCommand("alert", ["AAPL", "near", "200"], "/alert AAPL near 200")
        )
        assert "Unknown condition" in out

    async def test_create_alert_rejects_change_pct(self) -> None:
        server = self._server()
        out = await server._dispatch_bot_command(
            BotCommand("alert", ["AAPL", "change_pct", "5"], "/alert AAPL change_pct 5")
        )
        assert "change_pct" in out
        assert "CLI" in out

    async def test_create_alert_rejects_invalid_price(self) -> None:
        server = self._server()
        out = await server._dispatch_bot_command(
            BotCommand("alert", ["AAPL", "above", "abc"], "/alert AAPL above abc")
        )
        assert "Invalid price" in out

    async def test_create_alert_persists(self) -> None:
        monitor = _make_monitor()
        created = _alert("xx", "AAPL", 200)
        monitor.store.create.return_value = created
        server = self._server(monitor=monitor)
        out = await server._dispatch_bot_command(
            BotCommand("alert", ["AAPL", "above", "200"], "/alert AAPL above 200")
        )
        monitor.store.create.assert_called_once_with("AAPL", AlertCondition.ABOVE, 200.0)
        assert "Created alert xx" in out

    async def test_tasks_empty(self) -> None:
        executor = _make_executor()
        executor.store.get_active.return_value = []
        server = self._server(executor=executor)
        out = await server._dispatch_bot_command(BotCommand("tasks", [], "/tasks"))
        assert out == "No scheduled tasks."

    async def test_tasks_lists_each(self) -> None:
        executor = _make_executor()
        executor.store.get_active.return_value = [
            _task("t1", "AAPL"),
            _task("t2", "MSFT", schedule="daily 09:30"),
        ]
        server = self._server(executor=executor)
        out = await server._dispatch_bot_command(BotCommand("tasks", [], "/tasks"))
        assert "t1" in out
        assert "AAPL" in out
        assert "t2" in out
        assert "daily 09:30" in out

    async def test_schedule_validates_args(self) -> None:
        server = self._server()
        out = await server._dispatch_bot_command(
            BotCommand("schedule", ["analyze"], "/schedule analyze")
        )
        assert "Usage" in out

    async def test_schedule_rejects_unknown_action(self) -> None:
        server = self._server()
        out = await server._dispatch_bot_command(
            BotCommand(
                "schedule",
                ["foo", "AAPL", "every", "1h"],
                "/schedule foo AAPL every 1h",
            )
        )
        assert "Unknown action" in out

    async def test_schedule_rejects_invalid_spec(self) -> None:
        executor = _make_executor()
        executor.store.create.side_effect = ValueError("bad spec")
        server = self._server(executor=executor)
        out = await server._dispatch_bot_command(
            BotCommand(
                "schedule",
                ["analyze", "AAPL", "tomorrow"],
                "/schedule analyze AAPL tomorrow",
            )
        )
        assert "Invalid schedule" in out
        assert "bad spec" in out

    async def test_schedule_creates_task(self) -> None:
        executor = _make_executor()
        executor.store.create.return_value = _task("nn", "AAPL")
        server = self._server(executor=executor)
        out = await server._dispatch_bot_command(
            BotCommand(
                "schedule",
                ["analyze", "aapl", "every", "1h"],
                "/schedule analyze aapl every 1h",
            )
        )
        executor.store.create.assert_called_once_with(
            TaskActionType.ANALYZE, {"ticker": "AAPL"}, "every 1h"
        )
        assert "Scheduled task nn" in out


class TestNewBotCommands:
    """Tests for the /briefing, /watchlist, and /thesis handlers."""

    @staticmethod
    def _server(**kwargs) -> Server:
        return Server(_make_monitor(), _make_executor(), **kwargs)

    # ---- /briefing ----

    async def test_briefing_missing_deps_returns_hint(self) -> None:
        server = self._server()
        out = await server._dispatch_bot_command(BotCommand("briefing", [], "/briefing"))
        assert "Briefing unavailable" in out

    async def test_briefing_no_prior_session_returns_hint(self, tmp_path) -> None:
        import qracer.server as server_mod

        watchlist = MagicMock()
        data_registry = MagicMock()
        server = self._server(
            watchlist=watchlist,
            data_registry=data_registry,
            sessions_dir=tmp_path,
        )
        with patch.object(server_mod, "generate_briefing", AsyncMock(return_value=None)):
            out = await server._dispatch_bot_command(BotCommand("briefing", [], "/briefing"))
        assert "No briefing" in out

    async def test_briefing_returns_briefing_text(self, tmp_path) -> None:
        import qracer.server as server_mod

        watchlist = MagicMock()
        data_registry = MagicMock()
        server = self._server(
            watchlist=watchlist,
            data_registry=data_registry,
            sessions_dir=tmp_path,
        )
        with patch.object(
            server_mod,
            "generate_briefing",
            AsyncMock(return_value="Session Briefing\n  AAPL: $200"),
        ):
            out = await server._dispatch_bot_command(BotCommand("briefing", [], "/briefing"))
        assert "Session Briefing" in out
        assert "AAPL" in out

    async def test_briefing_failure_is_reported(self, tmp_path) -> None:
        import qracer.server as server_mod

        watchlist = MagicMock()
        data_registry = MagicMock()
        server = self._server(
            watchlist=watchlist,
            data_registry=data_registry,
            sessions_dir=tmp_path,
        )
        with patch.object(
            server_mod,
            "generate_briefing",
            AsyncMock(side_effect=RuntimeError("boom")),
        ):
            out = await server._dispatch_bot_command(BotCommand("briefing", [], "/briefing"))
        assert "Briefing failed" in out

    # ---- /watchlist ----

    async def test_watchlist_unconfigured(self) -> None:
        server = self._server()
        out = await server._dispatch_bot_command(BotCommand("watchlist", [], "/watchlist"))
        assert "unavailable" in out.lower()

    async def test_watchlist_empty(self) -> None:
        watchlist = MagicMock()
        watchlist.tickers = []
        server = self._server(watchlist=watchlist)
        out = await server._dispatch_bot_command(BotCommand("watchlist", [], "/watchlist"))
        assert "empty" in out.lower()

    async def test_watchlist_no_data_registry_shows_tickers_only(self) -> None:
        watchlist = MagicMock()
        watchlist.tickers = ["AAPL", "NVDA"]
        server = self._server(watchlist=watchlist)
        out = await server._dispatch_bot_command(BotCommand("watchlist", [], "/watchlist"))
        assert "AAPL" in out
        assert "NVDA" in out
        assert "$" not in out  # no price data

    async def test_watchlist_with_prices(self) -> None:
        watchlist = MagicMock()
        watchlist.tickers = ["AAPL", "NVDA"]
        data_registry = MagicMock()
        data_registry.async_get_with_fallback = AsyncMock(side_effect=[200.0, 1250.5])
        server = self._server(watchlist=watchlist, data_registry=data_registry)
        out = await server._dispatch_bot_command(BotCommand("watchlist", [], "/watchlist"))
        assert "AAPL: $200.00" in out
        assert "NVDA: $1,250.50" in out

    async def test_watchlist_handles_price_failures(self) -> None:
        watchlist = MagicMock()
        watchlist.tickers = ["AAPL", "BAD"]
        data_registry = MagicMock()
        data_registry.async_get_with_fallback = AsyncMock(
            side_effect=[200.0, RuntimeError("no feed")]
        )
        server = self._server(watchlist=watchlist, data_registry=data_registry)
        out = await server._dispatch_bot_command(BotCommand("watchlist", [], "/watchlist"))
        assert "AAPL: $200.00" in out
        assert "BAD: price unavailable" in out

    async def test_watchlist_handles_non_numeric_price(self) -> None:
        watchlist = MagicMock()
        watchlist.tickers = ["FOO"]
        data_registry = MagicMock()
        data_registry.async_get_with_fallback = AsyncMock(return_value=None)
        server = self._server(watchlist=watchlist, data_registry=data_registry)
        out = await server._dispatch_bot_command(BotCommand("watchlist", [], "/watchlist"))
        assert "FOO: price unavailable" in out

    # ---- /thesis ----

    async def test_thesis_no_reports_dir(self) -> None:
        server = self._server()
        out = await server._dispatch_bot_command(BotCommand("thesis", [], "/thesis"))
        assert "No saved theses" in out

    async def test_thesis_empty_reports_dir(self, tmp_path) -> None:
        server = self._server(reports_dir=tmp_path)
        out = await server._dispatch_bot_command(BotCommand("thesis", [], "/thesis"))
        assert "No saved theses" in out

    async def test_thesis_skips_reports_without_thesis_section(self, tmp_path) -> None:
        (tmp_path / "notes.md").write_text("# Hello\n\nJust a note.\n")
        server = self._server(reports_dir=tmp_path)
        out = await server._dispatch_bot_command(BotCommand("thesis", [], "/thesis"))
        assert "No saved theses" in out

    async def test_thesis_lists_recent_saved_reports(self, tmp_path) -> None:
        report = tmp_path / "AAPL-2026-04-15.md"
        report.write_text(
            "# Analysis Report: AAPL\n\n"
            "---\n\n"
            "## Trade Thesis\n\n"
            "- **Ticker:** AAPL\n"
            "- **Entry Zone:** $175.00 – $180.00\n"
            "- **Target Price:** $200.00\n"
            "- **Stop Loss:** $165.00\n"
            "\nLong AAPL on AI earnings.\n\n"
            "---\n\n"
            "## Data Sources\n\n- news\n"
        )
        server = self._server(reports_dir=tmp_path)
        out = await server._dispatch_bot_command(BotCommand("thesis", [], "/thesis"))
        assert "Recent theses" in out
        assert "AAPL-2026-04-15.md" in out
        assert "Entry Zone" in out
        # The "Data Sources" section should not leak into the thesis body.
        assert "Data Sources" not in out

    async def test_thesis_caps_at_three_entries(self, tmp_path) -> None:
        import time as _t

        body = "# Report\n\n## Trade Thesis\n\nSome thesis body.\n\n---\n"
        for i in range(5):
            p = tmp_path / f"T{i}.md"
            p.write_text(body)
            # Ensure distinct mtimes for deterministic ordering.
            mtime = 1_700_000_000 + i
            import os

            os.utime(p, (mtime, mtime))
            _ = _t  # silence unused-import in case of future refactors

        server = self._server(reports_dir=tmp_path)
        out = await server._dispatch_bot_command(BotCommand("thesis", [], "/thesis"))
        # Three most-recent (T4, T3, T2), oldest two excluded.
        assert "T4.md" in out
        assert "T3.md" in out
        assert "T2.md" in out
        assert "T0.md" not in out
        assert "T1.md" not in out


class TestChatIdRoutedReplies:
    async def test_reply_goes_back_to_sender_chat_id(self) -> None:
        monitor = _make_monitor()
        monitor.store.get_active.return_value = []
        executor = _make_executor()
        poller = _make_poller()
        poller.poll = AsyncMock(
            return_value=[
                BotCommand(
                    action="alerts",
                    args=[],
                    raw_text="/alerts",
                    chat_id="42",
                )
            ]
        )
        server = Server(monitor, executor, telegram_poller=poller)
        await server._tick()

        poller.send_reply.assert_awaited_once()
        args, kwargs = poller.send_reply.await_args
        # Reply text is first positional, chat_id is passed by keyword.
        assert kwargs.get("chat_id") == "42"

    async def test_reply_chat_id_is_none_when_command_has_no_chat_id(
        self,
    ) -> None:
        monitor = _make_monitor()
        monitor.store.get_active.return_value = []
        executor = _make_executor()
        poller = _make_poller()
        poller.poll = AsyncMock(
            return_value=[BotCommand(action="alerts", args=[], raw_text="/alerts")]
        )
        server = Server(monitor, executor, telegram_poller=poller)
        await server._tick()

        poller.send_reply.assert_awaited_once()
        _, kwargs = poller.send_reply.await_args
        # Falls back to the poller's primary chat.
        assert kwargs.get("chat_id") is None


class TestFormatDuration:
    def test_zero(self) -> None:
        assert _format_duration(0) == "0s"

    def test_seconds(self) -> None:
        assert _format_duration(45) == "45s"

    def test_minutes(self) -> None:
        assert _format_duration(125) == "2m 5s"

    def test_hours(self) -> None:
        assert _format_duration(3725) == "1h 2m 5s"

    def test_negative_clamped_to_zero(self) -> None:
        assert _format_duration(-5) == "0s"
