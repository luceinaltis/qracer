"""Tests for the Server loop."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

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


def _make_streaming_provider(connect_fails: bool = False):
    """Return a MagicMock mimicking the StreamingProvider protocol."""
    provider = MagicMock()
    provider.on_price = MagicMock()
    provider.on_news = MagicMock()
    provider.subscribe = AsyncMock(return_value=None)
    provider.unsubscribe = AsyncMock(return_value=None)
    provider.disconnect = AsyncMock(return_value=None)
    if connect_fails:
        provider.connect = AsyncMock(side_effect=ConnectionError("no network"))
    else:
        provider.connect = AsyncMock(return_value=None)
    return provider


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

    def test_help(self) -> None:
        server = self._server()
        out = server._dispatch_bot_command(BotCommand("help", [], "/help"))
        assert "/status" in out
        assert "/alerts" in out
        assert "/alert" in out
        assert "/tasks" in out
        assert "/schedule" in out

    def test_start_aliases_help(self) -> None:
        server = self._server()
        out = server._dispatch_bot_command(BotCommand("start", [], "/start"))
        assert "/status" in out

    def test_unknown_command(self) -> None:
        server = self._server()
        out = server._dispatch_bot_command(BotCommand("nope", [], "/nope"))
        assert "Unknown command" in out

    def test_analyze_returns_not_supported(self) -> None:
        server = self._server()
        out = server._dispatch_bot_command(BotCommand("analyze", ["AAPL"], "/analyze AAPL"))
        assert "not supported" in out.lower()

    def test_portfolio_returns_not_supported(self) -> None:
        server = self._server()
        out = server._dispatch_bot_command(BotCommand("portfolio", [], "/portfolio"))
        assert "not supported" in out.lower()

    def test_status(self) -> None:
        notifications = MagicMock()
        notifications.channels = ["telegram"]
        server = self._server(notifications=notifications)
        out = server._dispatch_bot_command(BotCommand("status", [], "/status"))
        assert "uptime" in out
        assert "telegram" in out
        assert "autonomous: off" in out

    def test_alerts_empty(self) -> None:
        monitor = _make_monitor()
        monitor.store.get_active.return_value = []
        server = self._server(monitor=monitor)
        out = server._dispatch_bot_command(BotCommand("alerts", [], "/alerts"))
        assert out == "No active alerts."

    def test_alerts_lists_each(self) -> None:
        monitor = _make_monitor()
        monitor.store.get_active.return_value = [
            _alert("a1", "AAPL", 200),
            _alert("b2", "MSFT", 410),
        ]
        server = self._server(monitor=monitor)
        out = server._dispatch_bot_command(BotCommand("alerts", [], "/alerts"))
        assert "a1" in out
        assert "AAPL" in out
        assert "b2" in out
        assert "MSFT" in out

    def test_create_alert_validates_args(self) -> None:
        server = self._server()
        out = server._dispatch_bot_command(BotCommand("alert", ["AAPL"], "/alert AAPL"))
        assert "Usage" in out

    def test_create_alert_rejects_unknown_condition(self) -> None:
        server = self._server()
        out = server._dispatch_bot_command(
            BotCommand("alert", ["AAPL", "near", "200"], "/alert AAPL near 200")
        )
        assert "Unknown condition" in out

    def test_create_alert_rejects_change_pct(self) -> None:
        server = self._server()
        out = server._dispatch_bot_command(
            BotCommand("alert", ["AAPL", "change_pct", "5"], "/alert AAPL change_pct 5")
        )
        assert "change_pct" in out
        assert "CLI" in out

    def test_create_alert_rejects_invalid_price(self) -> None:
        server = self._server()
        out = server._dispatch_bot_command(
            BotCommand("alert", ["AAPL", "above", "abc"], "/alert AAPL above abc")
        )
        assert "Invalid price" in out

    def test_create_alert_persists(self) -> None:
        monitor = _make_monitor()
        created = _alert("xx", "AAPL", 200)
        monitor.store.create.return_value = created
        server = self._server(monitor=monitor)
        out = server._dispatch_bot_command(
            BotCommand("alert", ["AAPL", "above", "200"], "/alert AAPL above 200")
        )
        monitor.store.create.assert_called_once_with("AAPL", AlertCondition.ABOVE, 200.0)
        assert "Created alert xx" in out

    def test_tasks_empty(self) -> None:
        executor = _make_executor()
        executor.store.get_active.return_value = []
        server = self._server(executor=executor)
        out = server._dispatch_bot_command(BotCommand("tasks", [], "/tasks"))
        assert out == "No scheduled tasks."

    def test_tasks_lists_each(self) -> None:
        executor = _make_executor()
        executor.store.get_active.return_value = [
            _task("t1", "AAPL"),
            _task("t2", "MSFT", schedule="daily 09:30"),
        ]
        server = self._server(executor=executor)
        out = server._dispatch_bot_command(BotCommand("tasks", [], "/tasks"))
        assert "t1" in out
        assert "AAPL" in out
        assert "t2" in out
        assert "daily 09:30" in out

    def test_schedule_validates_args(self) -> None:
        server = self._server()
        out = server._dispatch_bot_command(BotCommand("schedule", ["analyze"], "/schedule analyze"))
        assert "Usage" in out

    def test_schedule_rejects_unknown_action(self) -> None:
        server = self._server()
        out = server._dispatch_bot_command(
            BotCommand(
                "schedule",
                ["foo", "AAPL", "every", "1h"],
                "/schedule foo AAPL every 1h",
            )
        )
        assert "Unknown action" in out

    def test_schedule_rejects_invalid_spec(self) -> None:
        executor = _make_executor()
        executor.store.create.side_effect = ValueError("bad spec")
        server = self._server(executor=executor)
        out = server._dispatch_bot_command(
            BotCommand(
                "schedule",
                ["analyze", "AAPL", "tomorrow"],
                "/schedule analyze AAPL tomorrow",
            )
        )
        assert "Invalid schedule" in out
        assert "bad spec" in out

    def test_schedule_creates_task(self) -> None:
        executor = _make_executor()
        executor.store.create.return_value = _task("nn", "AAPL")
        server = self._server(executor=executor)
        out = server._dispatch_bot_command(
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


class TestServerStreaming:
    async def test_start_streaming_connects_and_subscribes(self) -> None:
        monitor = _make_monitor()
        monitor.store.get_active.return_value = [
            _alert("a1", "AAPL", 200),
            _alert("a2", "msft", 410),
        ]
        executor = _make_executor()
        provider = _make_streaming_provider()

        server = Server(monitor, executor, streaming_provider=provider)
        await server._start_streaming()

        provider.connect.assert_awaited_once()
        provider.on_price.assert_called_once()
        provider.subscribe.assert_awaited_once()
        args, _ = provider.subscribe.await_args
        assert sorted(args[0]) == ["AAPL", "MSFT"]
        assert server._streaming_active is True

    async def test_start_streaming_connect_failure_fallback(self) -> None:
        monitor = _make_monitor()
        monitor.store.get_active.return_value = []
        executor = _make_executor()
        provider = _make_streaming_provider(connect_fails=True)

        server = Server(monitor, executor, streaming_provider=provider)
        await server._start_streaming()  # must not raise

        assert server._streaming_active is False
        provider.subscribe.assert_not_called()

    async def test_realtime_price_triggers_alert(self) -> None:
        triggered_alert = _alert("a1", "AAPL", 200)
        monitor = _make_monitor()
        monitor.store.get_by_ticker.return_value = [triggered_alert]

        executor = _make_executor()
        notifications = MagicMock()
        notifications.channels = ["telegram"]
        notifications.notify = AsyncMock(return_value={"telegram": True})

        server = Server(monitor, executor, notifications)

        await server._on_realtime_price("AAPL", 205.0)

        monitor.store.mark_triggered.assert_called_once()
        notifications.notify.assert_called_once()
        notification = notifications.notify.call_args[0][0]
        assert "AAPL" in notification.title

    async def test_realtime_price_no_trigger_when_below_threshold(self) -> None:
        non_triggering = _alert("a1", "AAPL", 200)
        monitor = _make_monitor()
        monitor.store.get_by_ticker.return_value = [non_triggering]

        executor = _make_executor()
        server = Server(monitor, executor)

        await server._on_realtime_price("AAPL", 150.0)
        monitor.store.mark_triggered.assert_not_called()

    async def test_realtime_price_skips_inactive_alerts(self) -> None:
        inactive = _alert("a1", "AAPL", 200, active=False)
        monitor = _make_monitor()
        monitor.store.get_by_ticker.return_value = [inactive]

        executor = _make_executor()
        server = Server(monitor, executor)

        await server._on_realtime_price("AAPL", 250.0)
        monitor.store.mark_triggered.assert_not_called()

    async def test_tick_reconciles_streaming_subscriptions(self) -> None:
        monitor = _make_monitor()
        monitor.store.get_active.return_value = [_alert("a1", "TSLA", 250)]
        executor = _make_executor()
        provider = _make_streaming_provider()

        server = Server(monitor, executor, streaming_provider=provider)
        await server._start_streaming()
        provider.subscribe.reset_mock()

        # New alert added — reconcile should subscribe to the new ticker.
        monitor.store.get_active.return_value = [
            _alert("a1", "TSLA", 250),
            _alert("a2", "NVDA", 900),
        ]
        await server._tick()

        provider.subscribe.assert_awaited()
        new_symbols = provider.subscribe.await_args[0][0]
        assert "NVDA" in new_symbols
        assert "TSLA" not in new_symbols  # already subscribed

    async def test_stop_streaming_disconnects(self) -> None:
        monitor = _make_monitor()
        monitor.store.get_active.return_value = []
        executor = _make_executor()
        provider = _make_streaming_provider()

        server = Server(monitor, executor, streaming_provider=provider)
        await server._start_streaming()
        await server._stop_streaming()

        provider.disconnect.assert_awaited_once()
        assert server._streaming_active is False

    async def test_stop_streaming_noop_when_inactive(self) -> None:
        monitor = _make_monitor()
        executor = _make_executor()
        provider = _make_streaming_provider(connect_fails=True)

        server = Server(monitor, executor, streaming_provider=provider)
        await server._start_streaming()  # connect fails, stays inactive
        await server._stop_streaming()

        provider.disconnect.assert_not_called()


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
