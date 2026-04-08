"""Main Textual application for the qracer dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
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
    TasksPanel,
    WatchlistPanel,
)
from qracer.dashboard.sidebar import Sidebar

if TYPE_CHECKING:
    from qracer.alerts import AlertStore
    from qracer.data.registry import DataRegistry
    from qracer.tasks import TaskStore

# Map sidebar item IDs to panel classes.
_PANEL_MAP: dict[str, type] = {
    "overview": OverviewPanel,
    "portfolio": PortfolioPanel,
    "watchlist": WatchlistPanel,
    "alerts": AlertsPanel,
    "tasks": TasksPanel,
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
        ("5", "switch_panel('tasks')", "Tasks"),
    ]

    def __init__(
        self,
        *,
        data_registry: DataRegistry | None = None,
        alert_store: AlertStore | None = None,
        task_store: TaskStore | None = None,
    ) -> None:
        super().__init__()
        self._data_registry = data_registry
        self._alert_store = alert_store
        self._task_store = task_store

    def _build_panel(self, panel_cls: type, panel_id: str) -> Widget:
        """Instantiate a panel, injecting registries the panel accepts."""
        if panel_cls is OverviewPanel:
            return OverviewPanel(data_registry=self._data_registry, id=panel_id)
        if panel_cls is PortfolioPanel:
            return PortfolioPanel(data_registry=self._data_registry, id=panel_id)
        if panel_cls is WatchlistPanel:
            return WatchlistPanel(data_registry=self._data_registry, id=panel_id)
        if panel_cls is AlertsPanel:
            return AlertsPanel(alert_store=self._alert_store, id=panel_id)
        if panel_cls is TasksPanel:
            return TasksPanel(task_store=self._task_store, id=panel_id)
        widget: Widget = panel_cls(id=panel_id)
        return widget

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-layout"):
            yield Sidebar(id="sidebar")
            yield self._build_panel(OverviewPanel, "main-panel")
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
        new_panel = self._build_panel(panel_cls, "main-panel")
        old.remove()
        self.query_one("#main-layout").mount(new_panel)
        # Update sidebar highlight
        sidebar = self.query_one(Sidebar)
        sidebar.highlight_item(panel_id)
