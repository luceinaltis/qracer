"""Tests for the qracer dashboard TUI."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from qracer.cli import main
from qracer.dashboard.app import _PANEL_MAP, QracerDashboard
from qracer.dashboard.sidebar import Sidebar


class TestDashboardCLI:
    """Test the `qracer dashboard` CLI command is registered."""

    def test_dashboard_in_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "dashboard" in result.output

    def test_dashboard_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["dashboard", "--help"])
        assert result.exit_code == 0
        assert "Launch the interactive TUI dashboard" in result.output


class TestPanelMap:
    """Verify all sidebar items have a corresponding panel."""

    def test_all_sidebar_items_mapped(self) -> None:
        """Every item in the sidebar groups should have a panel class."""
        for _, items in Sidebar._GROUPS:
            for _, item_id in items:
                assert item_id in _PANEL_MAP, f"Missing panel for sidebar item: {item_id}"

    def test_panel_map_entries_are_widgets(self) -> None:
        for panel_id, cls in _PANEL_MAP.items():
            assert hasattr(cls, "compose"), f"{panel_id} panel must have compose()"


class TestSidebarGroups:
    """Verify sidebar structure matches the spec."""

    def test_has_three_groups(self) -> None:
        assert len(Sidebar._GROUPS) == 3

    def test_group_names(self) -> None:
        names = [g[0] for g in Sidebar._GROUPS]
        assert names == ["DASH", "CHAT", "SETTINGS"]

    def test_dash_group_items(self) -> None:
        dash_items = Sidebar._GROUPS[0][1]
        ids = [item_id for _, item_id in dash_items]
        assert ids == ["overview", "portfolio", "watchlist", "alerts", "tasks"]

    def test_chat_group_items(self) -> None:
        chat_items = Sidebar._GROUPS[1][1]
        ids = [item_id for _, item_id in chat_items]
        assert ids == ["new-chat", "history"]

    def test_settings_group_items(self) -> None:
        settings_items = Sidebar._GROUPS[2][1]
        ids = [item_id for _, item_id in settings_items]
        assert ids == ["general", "providers", "notifications"]


class TestAppConstruction:
    """Test the app can be instantiated without running."""

    def test_app_title(self) -> None:
        app = QracerDashboard()
        assert app.TITLE == "qracer"

    def test_css_path_exists(self) -> None:
        base = Path(__file__).parent.parent.parent / "qracer"
        css_path = base / "dashboard" / "dashboard.tcss"
        assert css_path.exists(), f"CSS file not found at {css_path}"
