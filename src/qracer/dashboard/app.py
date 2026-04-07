"""Main Textual application for the qracer dashboard."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, Header

from qracer.dashboard.panels import (
    AlertsPanel,
    GeneralSettingsPanel,
    HistoryPanel,
    NewChatPanel,
    NotificationsSettingsPanel,
    OverviewPanel,
    PortfolioPanel,
    ProvidersSettingsPanel,
    WatchlistPanel,
)
from qracer.dashboard.sidebar import Sidebar

# Map sidebar item IDs to panel classes.
_PANEL_MAP: dict[str, type] = {
    "overview": OverviewPanel,
    "portfolio": PortfolioPanel,
    "watchlist": WatchlistPanel,
    "alerts": AlertsPanel,
    "new-chat": NewChatPanel,
    "history": HistoryPanel,
    "general": GeneralSettingsPanel,
    "providers": ProvidersSettingsPanel,
    "notifications": NotificationsSettingsPanel,
}


class QracerDashboard(App):
    """Qracer terminal dashboard."""

    TITLE = "qracer"
    CSS_PATH = "dashboard.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("1", "switch_panel('overview')", "Overview"),
        ("2", "switch_panel('portfolio')", "Portfolio"),
        ("3", "switch_panel('watchlist')", "Watchlist"),
        ("4", "switch_panel('alerts')", "Alerts"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-layout"):
            yield Sidebar(id="sidebar")
            yield OverviewPanel(id="main-panel")
        yield Footer()

    def on_mount(self) -> None:
        """Select the default sidebar item after mount."""
        sidebar = self.query_one(Sidebar)
        sidebar.highlight_item("overview")

    def on_sidebar_item_selected(self, message: Sidebar.ItemSelected) -> None:
        """Handle sidebar navigation clicks."""
        self.action_switch_panel(message.item_id)

    def action_switch_panel(self, panel_id: str) -> None:
        """Replace the main panel with the selected one."""
        panel_cls = _PANEL_MAP.get(panel_id)
        if panel_cls is None:
            return
        old = self.query_one("#main-panel")
        new_panel = panel_cls(id="main-panel")
        old.remove()
        self.query_one("#main-layout").mount(new_panel)
        # Update sidebar highlight
        sidebar = self.query_one(Sidebar)
        sidebar.highlight_item(panel_id)
