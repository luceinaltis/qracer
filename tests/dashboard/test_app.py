"""Tests for the QracerDashboard app and sidebar interaction."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from qracer.config.models import QracerConfig
from qracer.dashboard.app import QracerDashboard
from qracer.dashboard.sidebar import Sidebar, _SidebarItem


class TestDashboardAppMount:
    @pytest.mark.asyncio
    async def test_app_mounts_with_overview(self) -> None:
        cfg = QracerConfig()
        with patch("qracer.dashboard.panels.load_config", return_value=cfg):
            async with QracerDashboard().run_test(size=(100, 30)) as pilot:
                sidebar = pilot.app.query_one(Sidebar)
                assert sidebar is not None
                panel = pilot.app.query_one("#main-panel")
                assert panel is not None

    @pytest.mark.asyncio
    async def test_switch_panel_invalid_id_is_noop(self) -> None:
        cfg = QracerConfig()
        with patch("qracer.dashboard.panels.load_config", return_value=cfg):
            async with QracerDashboard().run_test(size=(100, 30)) as pilot:
                pilot.app.action_switch_panel("nonexistent")
                await pilot.pause()
                # Should still have original panel
                panel = pilot.app.query_one("#main-panel")
                assert panel is not None


class TestSidebarWidget:
    @pytest.mark.asyncio
    async def test_sidebar_highlight_item(self) -> None:
        cfg = QracerConfig()
        with patch("qracer.dashboard.panels.load_config", return_value=cfg):
            async with QracerDashboard().run_test(size=(100, 30)) as pilot:
                sidebar = pilot.app.query_one(Sidebar)
                sidebar.highlight_item("watchlist")
                await pilot.pause()

                items = pilot.app.query(_SidebarItem)
                for item in items:
                    if item.item_id == "watchlist":
                        assert "--active" in item.classes
                    elif item.item_id == "overview":
                        assert "--active" not in item.classes

    @pytest.mark.asyncio
    async def test_sidebar_has_all_items(self) -> None:
        cfg = QracerConfig()
        with patch("qracer.dashboard.panels.load_config", return_value=cfg):
            async with QracerDashboard().run_test(size=(100, 30)) as pilot:
                items = pilot.app.query(_SidebarItem)
                item_ids = {item.item_id for item in items}
                expected = {
                    "overview",
                    "portfolio",
                    "watchlist",
                    "alerts",
                    "new-chat",
                    "history",
                    "general",
                    "providers",
                    "notifications",
                }
                assert item_ids == expected

    @pytest.mark.asyncio
    async def test_sidebar_initial_highlight_on_overview(self) -> None:
        cfg = QracerConfig()
        with patch("qracer.dashboard.panels.load_config", return_value=cfg):
            async with QracerDashboard().run_test(size=(100, 30)) as pilot:
                items = pilot.app.query(_SidebarItem)
                for item in items:
                    if item.item_id == "overview":
                        assert "--active" in item.classes
