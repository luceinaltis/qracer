"""Dashboard main-panel widgets for each sidebar section."""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import DataTable, Label, Static

from qracer.config.loader import _user_dir, load_config
from qracer.watchlist import Watchlist

logger = logging.getLogger(__name__)

REFRESH_INTERVAL_SECONDS = 5.0


def _watchlist_path() -> Path:
    return _user_dir() / "watchlist.json"


def _format_change(change: float) -> str:
    sign = "+" if change >= 0 else ""
    return f"{sign}{change:.2f}%"


# ---------------------------------------------------------------------------
# DASH panels
# ---------------------------------------------------------------------------


class OverviewPanel(VerticalScroll):
    """Portfolio summary + market overview."""

    DEFAULT_CSS = """
    OverviewPanel {
        padding: 1 2;
    }
    .panel-title {
        text-style: bold;
        padding-bottom: 1;
    }
    .card {
        border: solid $primary-background;
        padding: 1 2;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Overview", classes="panel-title")

        with Vertical(classes="card"):
            yield Label("Portfolio Summary", classes="panel-title")
            yield DataTable(id="overview-portfolio-table")

        with Vertical(classes="card"):
            yield Label("Market Overview", classes="panel-title")
            yield DataTable(id="overview-market-table")

    def on_mount(self) -> None:
        # Portfolio summary table
        ptable = self.query_one("#overview-portfolio-table", DataTable)
        ptable.add_columns("Ticker", "Shares", "Avg Cost", "Value", "P&L %")
        self._populate_portfolio(ptable)

        # Market overview table
        mtable = self.query_one("#overview-market-table", DataTable)
        mtable.add_columns("Index", "Value", "Change")
        self._populate_market(mtable)

        self.set_interval(REFRESH_INTERVAL_SECONDS, self._refresh)

    def _populate_portfolio(self, table: DataTable) -> None:
        table.clear()
        cfg = load_config()
        if not cfg.portfolio.holdings:
            table.add_row("—", "—", "—", "—", "—")
            return
        for h in cfg.portfolio.holdings:
            value = h.shares * h.avg_cost
            table.add_row(
                h.ticker,
                f"{h.shares:.2f}",
                f"${h.avg_cost:,.2f}",
                f"${value:,.2f}",
                "—",
            )

    def _populate_market(self, table: DataTable) -> None:
        table.clear()
        # Static placeholder — real-time data requires data registry integration.
        for name in ("S&P 500", "NASDAQ", "VIX"):
            table.add_row(name, "—", "—")

    def _refresh(self) -> None:
        ptable = self.query_one("#overview-portfolio-table", DataTable)
        self._populate_portfolio(ptable)


class PortfolioPanel(VerticalScroll):
    """Holdings detail with P&L, allocation weight, and risk metrics."""

    DEFAULT_CSS = """
    PortfolioPanel { padding: 1 2; }
    .panel-title { text-style: bold; padding-bottom: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Label("Portfolio", classes="panel-title")
        yield DataTable(id="portfolio-table")

    def on_mount(self) -> None:
        table = self.query_one("#portfolio-table", DataTable)
        table.add_columns("Ticker", "Shares", "Avg Cost", "Value", "Weight", "P&L %")
        self._populate(table)
        self.set_interval(REFRESH_INTERVAL_SECONDS, self._refresh)

    def _populate(self, table: DataTable) -> None:
        table.clear()
        cfg = load_config()
        holdings = cfg.portfolio.holdings
        if not holdings:
            table.add_row("—", "—", "—", "—", "—", "—")
            return

        total = sum(h.shares * h.avg_cost for h in holdings)
        for h in holdings:
            value = h.shares * h.avg_cost
            weight = (value / total * 100) if total > 0 else 0.0
            table.add_row(
                h.ticker,
                f"{h.shares:.2f}",
                f"${h.avg_cost:,.2f}",
                f"${value:,.2f}",
                f"{weight:.1f}%",
                "—",
            )

    def _refresh(self) -> None:
        table = self.query_one("#portfolio-table", DataTable)
        self._populate(table)


class WatchlistPanel(VerticalScroll):
    """Watchlist with tickers and status."""

    DEFAULT_CSS = """
    WatchlistPanel { padding: 1 2; }
    .panel-title { text-style: bold; padding-bottom: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Label("Watchlist", classes="panel-title")
        yield DataTable(id="watchlist-table")
        yield Label("", id="watchlist-hint")

    def on_mount(self) -> None:
        table = self.query_one("#watchlist-table", DataTable)
        table.add_columns("Ticker", "Price", "Change %")
        self._populate(table)
        self.set_interval(REFRESH_INTERVAL_SECONDS, self._refresh)

    def _populate(self, table: DataTable) -> None:
        table.clear()
        wl = Watchlist(_watchlist_path())
        hint = self.query_one("#watchlist-hint", Label)
        if not wl.tickers:
            hint.update("Watchlist is empty. Use 'qracer repl' → 'watch TICKER' to add.")
            return
        hint.update("")
        for ticker in wl.tickers:
            table.add_row(ticker, "—", "—")

    def _refresh(self) -> None:
        table = self.query_one("#watchlist-table", DataTable)
        self._populate(table)


class AlertsPanel(VerticalScroll):
    """Active alerts and trigger history."""

    DEFAULT_CSS = """
    AlertsPanel { padding: 1 2; }
    .panel-title { text-style: bold; padding-bottom: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Label("Alerts", classes="panel-title")
        yield Static("No active alerts. Price alert support coming soon (see issue #45).")


# ---------------------------------------------------------------------------
# CHAT panels
# ---------------------------------------------------------------------------


class NewChatPanel(VerticalScroll):
    """Placeholder for starting a new chat session."""

    DEFAULT_CSS = """
    NewChatPanel { padding: 1 2; }
    .panel-title { text-style: bold; padding-bottom: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Label("New Chat", classes="panel-title")
        yield Static(
            "Start a new conversation session.\n\n"
            "Use 'qracer repl' for the full conversational agent experience.\n"
            "Dashboard chat integration is planned for a future release."
        )


class HistoryPanel(VerticalScroll):
    """Show past conversation sessions."""

    DEFAULT_CSS = """
    HistoryPanel { padding: 1 2; }
    .panel-title { text-style: bold; padding-bottom: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Label("Chat History", classes="panel-title")
        yield DataTable(id="history-table")

    def on_mount(self) -> None:
        table = self.query_one("#history-table", DataTable)
        table.add_columns("Session ID", "Date", "Queries")
        self._populate(table)

    def _populate(self, table: DataTable) -> None:
        table.clear()
        sessions_dir = _user_dir() / "sessions"
        if not sessions_dir.is_dir():
            return
        session_files = sorted(sessions_dir.glob("*.jsonl"), reverse=True)[:20]
        if not session_files:
            return
        for f in session_files:
            sid = f.stem
            mtime = f.stat().st_mtime
            dt = datetime.datetime.fromtimestamp(mtime)
            line_count = sum(1 for _ in f.open())
            table.add_row(sid, dt.strftime("%Y-%m-%d %H:%M"), str(line_count))


# ---------------------------------------------------------------------------
# SETTINGS panels
# ---------------------------------------------------------------------------


class GeneralSettingsPanel(VerticalScroll):
    """Display general application settings."""

    DEFAULT_CSS = """
    GeneralSettingsPanel { padding: 1 2; }
    .panel-title { text-style: bold; padding-bottom: 1; }
    .setting-row { padding: 0 0 0 1; height: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Label("General Settings", classes="panel-title")
        yield DataTable(id="general-settings-table")

    def on_mount(self) -> None:
        table = self.query_one("#general-settings-table", DataTable)
        table.add_columns("Setting", "Value")
        cfg = load_config()
        table.add_row("Language", cfg.app.language)
        table.add_row("Default Mode", cfg.app.default_mode)
        table.add_row("LLM Provider", cfg.app.llm_provider)
        table.add_row("LLM Model", cfg.app.llm_model)
        table.add_row("Currency", cfg.portfolio.currency)


class ProvidersSettingsPanel(VerticalScroll):
    """Display provider configuration."""

    DEFAULT_CSS = """
    ProvidersSettingsPanel { padding: 1 2; }
    .panel-title { text-style: bold; padding-bottom: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Label("Providers", classes="panel-title")
        yield DataTable(id="providers-table")

    def on_mount(self) -> None:
        table = self.query_one("#providers-table", DataTable)
        table.add_columns("Provider", "Kind", "Enabled", "Tier", "Priority", "API Key Env")
        cfg = load_config()
        for name, prov in cfg.providers.providers.items():
            table.add_row(
                name,
                prov.kind,
                "✓" if prov.enabled else "✗",
                prov.tier,
                str(prov.priority),
                prov.api_key_env or "—",
            )


class NotificationsSettingsPanel(VerticalScroll):
    """Notification channel settings placeholder."""

    DEFAULT_CSS = """
    NotificationsSettingsPanel { padding: 1 2; }
    .panel-title { text-style: bold; padding-bottom: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Label("Notifications", classes="panel-title")
        yield Static(
            "Notification channels are not yet configured.\n\n"
            "Telegram integration is planned (see issue #38).\n"
            "Price alert thresholds are planned (see issue #45)."
        )
