"""Server — long-running service loop for qracer serve.

Drives TaskExecutor and AlertMonitor without user input, sending
notifications via the NotificationRegistry when events occur.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path

from qracer.alert_monitor import AlertMonitor
from qracer.alerts import AlertCondition
from qracer.autonomous import AutonomousMonitor
from qracer.conversation.quickpath import generate_briefing
from qracer.data.providers import PriceProvider
from qracer.data.registry import DataRegistry
from qracer.notifications.providers import Notification, NotificationCategory
from qracer.notifications.registry import NotificationRegistry
from qracer.notifications.telegram_poller import BotCommand, TelegramBotPoller
from qracer.task_executor import TaskExecutor
from qracer.tasks import TaskActionType
from qracer.watchlist import Watchlist

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
        telegram_poller: TelegramBotPoller | None = None,
        tick_interval: float = 1.0,
        watchlist: Watchlist | None = None,
        data_registry: DataRegistry | None = None,
        sessions_dir: Path | None = None,
        reports_dir: Path | None = None,
    ) -> None:
        self._alert_monitor = alert_monitor
        self._task_executor = task_executor
        self._autonomous_monitor = autonomous_monitor
        self._notifications = notifications or NotificationRegistry()
        self._telegram_poller = telegram_poller
        self._tick_interval = tick_interval
        self._watchlist = watchlist
        self._data_registry = data_registry
        self._sessions_dir = sessions_dir
        self._reports_dir = reports_dir
        self._shutdown_event = asyncio.Event()
        self._started_at: float | None = None

    async def run(self) -> None:
        """Main loop — runs until shutdown() is called."""
        logger.info("Server started (tick=%.1fs)", self._tick_interval)
        self._started_at = time.monotonic()

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

        if self._telegram_poller is not None:
            try:
                commands = await self._telegram_poller.poll()
            except Exception:
                logger.debug("Telegram poll failed", exc_info=True)
                commands = []
            for command in commands:
                await self._handle_bot_command(command)

    async def _handle_bot_command(self, command: BotCommand) -> None:
        """Dispatch an incoming bot command and reply with the result."""
        try:
            reply = await self._dispatch_bot_command(command)
        except Exception as exc:
            logger.exception("Bot command handler failed: /%s", command.action)
            reply = f"Error handling /{command.action}: {exc}"
        if reply and self._telegram_poller is not None:
            await self._telegram_poller.send_reply(
                reply, chat_id=command.chat_id or None
            )

    async def _dispatch_bot_command(self, command: BotCommand) -> str:
        """Route a :class:`BotCommand` to the matching handler.

        Handlers return the reply text to send back to the user. Long
        replies are truncated by the poller before transmission.
        """
        action = command.action
        if action in {"help", "start"}:
            return self._cmd_help()
        if action == "status":
            return self._cmd_status()
        if action == "alerts":
            return self._cmd_alerts()
        if action == "alert":
            return self._cmd_create_alert(command.args)
        if action == "tasks":
            return self._cmd_tasks()
        if action == "schedule":
            return self._cmd_schedule(command.args)
        if action == "briefing":
            return await self._cmd_briefing()
        if action == "watchlist":
            return await self._cmd_watchlist()
        if action == "thesis":
            return self._cmd_thesis()
        if action in {"analyze", "portfolio"}:
            return (
                f"/{action} is not supported in bot mode yet — "
                "use the qracer CLI on the host. Try /help."
            )
        return f"Unknown command: /{action}. Try /help."

    # ------------------------------------------------------------------
    # Individual command handlers
    # ------------------------------------------------------------------

    @staticmethod
    def _cmd_help() -> str:
        return (
            "qracer bot commands:\n"
            "/status — server status and uptime\n"
            "/briefing — session briefing since the last REPL run\n"
            "/watchlist — watchlist tickers with current prices\n"
            "/thesis — recent saved trade theses\n"
            "/alerts — list active price alerts\n"
            "/alert TICKER above|below PRICE — create a price alert\n"
            "/tasks — list scheduled tasks\n"
            "/schedule ACTION TICKER SCHEDULE — schedule a task\n"
            "    e.g. /schedule analyze AAPL every 1h\n"
            "/help — show this message"
        )

    def _cmd_status(self) -> str:
        uptime = "unknown"
        if self._started_at is not None:
            uptime = _format_duration(time.monotonic() - self._started_at)
        channels = ", ".join(self._notifications.channels) or "none"
        autonomous = "on" if self._autonomous_monitor else "off"
        return (
            "qracer status\n"
            f"  uptime: {uptime}\n"
            f"  notifications: {channels}\n"
            f"  autonomous: {autonomous}"
        )

    def _cmd_alerts(self) -> str:
        alerts = self._alert_monitor.store.get_active()
        if not alerts:
            return "No active alerts."
        lines = ["Active alerts:"]
        for a in alerts:
            lines.append(f"  {a.id}  {a.describe()}")
        return "\n".join(lines)

    def _cmd_create_alert(self, args: list[str]) -> str:
        if len(args) < 3:
            return "Usage: /alert TICKER above|below PRICE  (e.g. /alert AAPL above 200)"
        ticker, condition_str, price_str = args[0], args[1].lower(), args[2]
        try:
            condition = AlertCondition(condition_str)
        except ValueError:
            return f"Unknown condition '{condition_str}'. Use 'above' or 'below'."
        if condition is AlertCondition.CHANGE_PCT:
            return "Use 'above' or 'below' from the bot — change_pct alerts need the CLI."
        try:
            threshold = float(price_str)
        except ValueError:
            return f"Invalid price '{price_str}' — must be a number."
        alert = self._alert_monitor.store.create(ticker, condition, threshold)
        return f"Created alert {alert.id}: {alert.describe()}"

    def _cmd_tasks(self) -> str:
        tasks = self._task_executor.store.get_active()
        if not tasks:
            return "No scheduled tasks."
        lines = ["Scheduled tasks:"]
        for t in tasks:
            lines.append(f"  {t.id}  {t.describe()}")
        return "\n".join(lines)

    def _cmd_schedule(self, args: list[str]) -> str:
        if len(args) < 3:
            return (
                "Usage: /schedule ACTION TICKER SCHEDULE\n"
                "  ACTION: analyze | news_scan | portfolio_snapshot\n"
                "  e.g. /schedule analyze AAPL every 1h"
            )
        action_str = args[0].lower()
        ticker = args[1].upper()
        schedule_spec = " ".join(args[2:])
        try:
            action_type = TaskActionType(action_str)
        except ValueError:
            valid = ", ".join(t.value for t in TaskActionType)
            return f"Unknown action '{action_str}'. Valid: {valid}"
        try:
            task = self._task_executor.store.create(action_type, {"ticker": ticker}, schedule_spec)
        except ValueError as exc:
            return f"Invalid schedule: {exc}"
        return f"Scheduled task {task.id}: {task.describe()}"

    async def _cmd_briefing(self) -> str:
        """Compose a session-start-style briefing from current state."""
        if (
            self._watchlist is None
            or self._data_registry is None
            or self._sessions_dir is None
        ):
            return (
                "Briefing unavailable in this mode. "
                "Run `qracer repl` on the host for session-start briefings."
            )
        try:
            briefing = await generate_briefing(
                self._watchlist,
                self._data_registry,
                self._alert_monitor.store,
                self._task_executor.store,
                self._sessions_dir,
            )
        except Exception:
            logger.exception("Telegram /briefing generation failed")
            return "Briefing failed — see server logs for details."
        if not briefing:
            return "No briefing: no prior session on file (or nothing new since)."
        return briefing

    async def _cmd_watchlist(self) -> str:
        """Return watchlist tickers with current prices."""
        if self._watchlist is None:
            return "Watchlist unavailable — not configured on this server."
        tickers = self._watchlist.tickers
        if not tickers:
            return "Watchlist is empty. Add tickers from the qracer REPL with 'watch TICKER'."
        if self._data_registry is None:
            return "Watchlist:\n" + "\n".join(f"  {t}" for t in tickers)
        lines = [f"Watchlist ({len(tickers)}):"]
        for ticker in tickers:
            try:
                price = await self._data_registry.async_get_with_fallback(
                    PriceProvider, "get_price", ticker
                )
            except Exception:
                logger.debug("Price fetch failed for %s", ticker, exc_info=True)
                lines.append(f"  {ticker}: price unavailable")
                continue
            if isinstance(price, (int, float)):
                lines.append(f"  {ticker}: ${price:,.2f}")
            else:
                lines.append(f"  {ticker}: price unavailable")
        return "\n".join(lines)

    def _cmd_thesis(self) -> str:
        """Summarise the most recent saved trade-thesis report(s).

        Reads the ``reports_dir`` the REPL writes to (via
        :class:`~qracer.conversation.report_exporter.ReportExporter`) and
        extracts the Trade-Thesis section of each Markdown report.
        """
        if self._reports_dir is None or not self._reports_dir.exists():
            return (
                "No saved theses found. Run the qracer REPL and use "
                "`save` after a thesis query to make theses visible here."
            )
        try:
            md_files = sorted(
                (p for p in self._reports_dir.glob("*.md") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            logger.debug("Failed to list reports dir", exc_info=True)
            return "Thesis lookup failed — see server logs for details."

        entries: list[str] = []
        for path in md_files:
            if len(entries) >= 3:
                break
            summary = _extract_thesis_section(path)
            if summary is None:
                continue
            when = time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)
            )
            entries.append(f"[{when}] {path.name}\n{summary}")

        if not entries:
            return "No saved theses found in reports directory."
        header = f"Recent theses ({len(entries)}):"
        return "\n\n".join([header, *entries])

    async def _notify(self, category: NotificationCategory, title: str, body: str) -> None:
        """Send a notification if any channels are registered."""
        if not self._notifications.channels:
            return
        notification = Notification(category=category, title=title, body=body)
        await self._notifications.notify(notification)

    def shutdown(self) -> None:
        """Signal the server to stop after the current tick."""
        self._shutdown_event.set()


_THESIS_HEADING = re.compile(r"^##\s+Trade Thesis\s*$", re.MULTILINE)


def _extract_thesis_section(path: Path) -> str | None:
    """Return the ``## Trade Thesis`` body from a saved Markdown report.

    Returns ``None`` if the file is unreadable or has no thesis section.
    The returned text is trimmed to the next ``---`` or ``##`` boundary
    and capped at 800 characters to keep Telegram replies compact.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _THESIS_HEADING.search(text)
    if match is None:
        return None
    body = text[match.end() :]

    # Stop at the next top-level section or horizontal rule.
    stop = len(body)
    for marker in ("\n## ", "\n---"):
        idx = body.find(marker)
        if idx >= 0 and idx < stop:
            stop = idx
    body = body[:stop].strip()
    if len(body) > 800:
        body = body[:797] + "..."
    return body or None


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as ``"1h 23m 45s"`` (omitting empty units)."""
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)
