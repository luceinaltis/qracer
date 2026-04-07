"""Server — long-running service loop for qracer serve.

Drives TaskExecutor and AlertMonitor without user input, sending
notifications via the NotificationRegistry when events occur.
"""

from __future__ import annotations

import asyncio
import logging

from qracer.alert_monitor import AlertMonitor
from qracer.autonomous import AutonomousMonitor
from qracer.notifications.providers import Notification, NotificationCategory
from qracer.notifications.registry import NotificationRegistry
from qracer.task_executor import TaskExecutor

logger = logging.getLogger(__name__)


class Server:
    """Headless service loop — replaces the REPL's input()-driven heartbeat.

    Usage::

        server = Server(alert_monitor, task_executor, notifications)
        await server.run()       # blocks until shutdown() is called
        server.shutdown()        # from a signal handler
    """

    def __init__(
        self,
        alert_monitor: AlertMonitor,
        task_executor: TaskExecutor,
        notifications: NotificationRegistry | None = None,
        *,
        autonomous_monitor: AutonomousMonitor | None = None,
        tick_interval: float = 1.0,
    ) -> None:
        self._alert_monitor = alert_monitor
        self._task_executor = task_executor
        self._autonomous_monitor = autonomous_monitor
        self._notifications = notifications or NotificationRegistry()
        self._tick_interval = tick_interval
        self._shutdown_event = asyncio.Event()

    async def run(self) -> None:
        """Main loop — runs until shutdown() is called."""
        logger.info("Server started (tick=%.1fs)", self._tick_interval)

        while not self._shutdown_event.is_set():
            await self._tick()
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._tick_interval,
                )
            except asyncio.TimeoutError:
                pass

        logger.info("Server stopped")

    async def _tick(self) -> None:
        """Single heartbeat — check alerts and tasks."""
        if self._alert_monitor.should_check():
            try:
                triggered = await self._alert_monitor.check()
                for result in triggered:
                    logger.info("Alert triggered: %s", result.message)
                    await self._notify(
                        NotificationCategory.PRICE_ALERT,
                        result.message,
                        result.message,
                    )
            except Exception:
                logger.debug("Alert check failed", exc_info=True)

        if self._task_executor.should_check():
            try:
                results = await self._task_executor.check()
                for r in results:
                    if r.success:
                        logger.info("Task completed: %s", r.task.describe())
                    else:
                        logger.warning("Task failed: %s — %s", r.task.describe(), r.error)
                        await self._notify(
                            NotificationCategory.AUTONOMOUS_MODE,
                            f"Task failed: {r.task.describe()}",
                            r.error or "unknown error",
                        )
            except Exception:
                logger.debug("Task check failed", exc_info=True)

        if self._autonomous_monitor and self._autonomous_monitor.should_check():
            try:
                auto_alerts = await self._autonomous_monitor.check()
                for alert in auto_alerts:
                    logger.info("Autonomous alert: %s", alert.summary)
                    await self._notify(
                        NotificationCategory.AUTONOMOUS_MODE,
                        f"[{alert.severity.value.upper()}] {alert.ticker}",
                        alert.summary,
                    )
            except Exception:
                logger.debug("Autonomous check failed", exc_info=True)

    async def _notify(self, category: NotificationCategory, title: str, body: str) -> None:
        """Send a notification if any channels are registered."""
        if not self._notifications.channels:
            return
        notification = Notification(category=category, title=title, body=body)
        await self._notifications.notify(notification)

    def shutdown(self) -> None:
        """Signal the server to stop after the current tick."""
        self._shutdown_event.set()
