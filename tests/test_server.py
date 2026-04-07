"""Tests for the Server loop."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from qracer.server import Server


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
