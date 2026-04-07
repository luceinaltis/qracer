"""Sidebar widget with grouped navigation items."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Label, Static


class _SidebarItem(Static):
    """A clickable sidebar menu item."""

    DEFAULT_CSS = """
    _SidebarItem {
        padding: 0 1;
        height: 1;
        content-align-horizontal: left;
    }
    _SidebarItem:hover {
        background: $surface-lighten-2;
    }
    _SidebarItem.--active {
        background: $accent;
        color: $text;
    }
    """

    def __init__(self, label: str, item_id: str) -> None:
        super().__init__(label)
        self.item_id = item_id

    def on_click(self) -> None:
        self.post_message(Sidebar.ItemSelected(self.item_id))


class Sidebar(Widget):
    """Left sidebar with grouped navigation."""

    DEFAULT_CSS = """
    Sidebar {
        width: 22;
        dock: left;
        background: $surface;
        border-right: solid $primary-background;
    }
    .sidebar-group-label {
        color: $text-muted;
        padding: 1 1 0 1;
        text-style: bold;
    }
    """

    class ItemSelected(Message):
        """Posted when a sidebar item is clicked."""

        def __init__(self, item_id: str) -> None:
            super().__init__()
            self.item_id = item_id

    _GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
        (
            "DASH",
            [
                ("Overview", "overview"),
                ("Portfolio", "portfolio"),
                ("Watchlist", "watchlist"),
                ("Alerts", "alerts"),
            ],
        ),
        (
            "CHAT",
            [
                ("New Chat", "new-chat"),
                ("History", "history"),
            ],
        ),
        (
            "SETTINGS",
            [
                ("General", "general"),
                ("Providers", "providers"),
                ("Notifications", "notifications"),
            ],
        ),
    ]

    def compose(self) -> ComposeResult:
        for group_label, items in self._GROUPS:
            yield Label(group_label, classes="sidebar-group-label")
            with Vertical():
                for display_label, item_id in items:
                    yield _SidebarItem(f"  {display_label}", item_id)

    def highlight_item(self, active_id: str) -> None:
        """Highlight the active item, removing previous highlight."""
        for item in self.query(_SidebarItem):
            if item.item_id == active_id:
                item.add_class("--active")
            else:
                item.remove_class("--active")
