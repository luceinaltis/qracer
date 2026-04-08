"""Tests for dashboard panel widgets using Textual's async testing."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Label

from qracer.alerts import AlertCondition, AlertStore
from qracer.config.models import (
    Holding,
    PortfolioConfig,
    ProviderConfig,
    ProvidersConfig,
    QracerConfig,
)
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
    _format_change,
    _format_next_run,
    _watchlist_path,
)
from qracer.tasks import TaskActionType, TaskStore


def _make_config(
    holdings: list[Holding] | None = None,
    providers: dict[str, ProviderConfig] | None = None,
) -> QracerConfig:
    return QracerConfig(
        portfolio=PortfolioConfig(holdings=holdings or []),
        providers=ProvidersConfig(providers=providers or {}),
    )


class _FakeRegistry:
    """Minimal async-get-with-fallback stand-in for DataRegistry."""

    def __init__(self, prices: dict[str, float]) -> None:
        self._prices = prices
        self.calls: list[str] = []

    async def async_get_with_fallback(
        self, capability: Any, method: str, *args: Any, **kwargs: Any
    ) -> Any:
        ticker = args[0]
        self.calls.append(ticker)
        if ticker in self._prices:
            return self._prices[ticker]
        raise RuntimeError(f"no price for {ticker}")


class _SinglePanelApp(App):
    """Test harness that mounts a single panel."""

    def __init__(self, panel_cls: type, panel_id: str = "test-panel", **kwargs: Any) -> None:
        super().__init__()
        self._panel_cls = panel_cls
        self._panel_id = panel_id
        self._panel_kwargs = kwargs

    def compose(self) -> ComposeResult:
        yield self._panel_cls(id=self._panel_id, **self._panel_kwargs)


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestFormatChange:
    def test_positive(self) -> None:
        assert _format_change(1.5) == "+1.50%"

    def test_negative(self) -> None:
        assert _format_change(-2.3) == "-2.30%"

    def test_zero(self) -> None:
        assert _format_change(0.0) == "+0.00%"


class TestWatchlistPath:
    def test_returns_path(self, tmp_path: Path) -> None:
        with patch("qracer.dashboard.panels._user_dir", return_value=tmp_path):
            result = _watchlist_path()
        assert result == tmp_path / "watchlist.json"


class TestFormatNextRun:
    def test_none(self) -> None:
        assert _format_next_run(None) == "—"

    def test_past_is_due(self) -> None:
        past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        assert _format_next_run(past) == "due"

    def test_future_minutes(self) -> None:
        future = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
        assert _format_next_run(future).startswith("in ")
        assert "m" in _format_next_run(future)

    def test_future_hours(self) -> None:
        future = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
        assert _format_next_run(future).endswith("h")

    def test_invalid_string_returned_verbatim(self) -> None:
        assert _format_next_run("not-a-date") == "not-a-date"


# ---------------------------------------------------------------------------
# OverviewPanel
# ---------------------------------------------------------------------------


class TestOverviewPanel:
    @pytest.mark.asyncio
    async def test_overview_empty_portfolio(self, tmp_path: Path) -> None:
        cfg = _make_config()
        with (
            patch("qracer.dashboard.panels.load_config", return_value=cfg),
            patch("qracer.dashboard.panels._watchlist_path", return_value=tmp_path / "wl.json"),
        ):
            async with _SinglePanelApp(OverviewPanel).run_test(size=(80, 24)) as pilot:
                table = pilot.app.query_one("#overview-portfolio-table", DataTable)
                assert table.row_count == 1  # placeholder row

    @pytest.mark.asyncio
    async def test_overview_with_holdings(self, tmp_path: Path) -> None:
        holdings = [
            Holding(ticker="AAPL", shares=10, avg_cost=150.0),
            Holding(ticker="TSLA", shares=5, avg_cost=200.0),
        ]
        cfg = _make_config(holdings=holdings)
        with (
            patch("qracer.dashboard.panels.load_config", return_value=cfg),
            patch("qracer.dashboard.panels._watchlist_path", return_value=tmp_path / "wl.json"),
        ):
            async with _SinglePanelApp(OverviewPanel).run_test(size=(80, 24)) as pilot:
                table = pilot.app.query_one("#overview-portfolio-table", DataTable)
                assert table.row_count == 2

    @pytest.mark.asyncio
    async def test_overview_watchlist_table(self, tmp_path: Path) -> None:
        wl_path = tmp_path / "watchlist.json"
        wl_path.write_text(json.dumps(["AAPL", "TSLA"]))
        cfg = _make_config()
        with (
            patch("qracer.dashboard.panels.load_config", return_value=cfg),
            patch("qracer.dashboard.panels._watchlist_path", return_value=wl_path),
        ):
            async with _SinglePanelApp(OverviewPanel).run_test(size=(80, 24)) as pilot:
                table = pilot.app.query_one("#overview-watchlist-table", DataTable)
                assert table.row_count == 2

    @pytest.mark.asyncio
    async def test_overview_refresh_data_populates_prices(self, tmp_path: Path) -> None:
        holdings = [Holding(ticker="AAPL", shares=10, avg_cost=150.0)]
        cfg = _make_config(holdings=holdings)
        wl_path = tmp_path / "watchlist.json"
        wl_path.write_text(json.dumps(["TSLA"]))
        fake = _FakeRegistry({"AAPL": 175.0, "TSLA": 250.0})

        with (
            patch("qracer.dashboard.panels.load_config", return_value=cfg),
            patch("qracer.dashboard.panels._watchlist_path", return_value=wl_path),
        ):
            async with _SinglePanelApp(OverviewPanel, data_registry=fake).run_test(
                size=(80, 24)
            ) as pilot:
                panel = pilot.app.query_one(OverviewPanel)
                await panel.refresh_data()
                # Both tickers fetched
                assert set(fake.calls) == {"AAPL", "TSLA"}
                assert panel._prices == {"AAPL": 175.0, "TSLA": 250.0}


# ---------------------------------------------------------------------------
# PortfolioPanel
# ---------------------------------------------------------------------------


class TestPortfolioPanel:
    @pytest.mark.asyncio
    async def test_portfolio_empty(self) -> None:
        cfg = _make_config()
        with patch("qracer.dashboard.panels.load_config", return_value=cfg):
            async with _SinglePanelApp(PortfolioPanel).run_test(size=(80, 24)) as pilot:
                table = pilot.app.query_one("#portfolio-table", DataTable)
                assert table.row_count == 1  # placeholder

    @pytest.mark.asyncio
    async def test_portfolio_with_holdings(self) -> None:
        holdings = [
            Holding(ticker="AAPL", shares=10, avg_cost=150.0),
            Holding(ticker="MSFT", shares=20, avg_cost=300.0),
        ]
        cfg = _make_config(holdings=holdings)
        with patch("qracer.dashboard.panels.load_config", return_value=cfg):
            async with _SinglePanelApp(PortfolioPanel).run_test(size=(80, 24)) as pilot:
                table = pilot.app.query_one("#portfolio-table", DataTable)
                assert table.row_count == 2

    @pytest.mark.asyncio
    async def test_portfolio_refresh_builds_snapshot(self) -> None:
        holdings = [
            Holding(ticker="AAPL", shares=10, avg_cost=150.0),
            Holding(ticker="MSFT", shares=20, avg_cost=300.0),
        ]
        cfg = _make_config(holdings=holdings)
        # AAPL up 20%, MSFT flat
        fake = _FakeRegistry({"AAPL": 180.0, "MSFT": 300.0})

        with patch("qracer.dashboard.panels.load_config", return_value=cfg):
            async with _SinglePanelApp(PortfolioPanel, data_registry=fake).run_test(
                size=(100, 30)
            ) as pilot:
                panel = pilot.app.query_one(PortfolioPanel)
                await panel.refresh_data()
                # Prices cached
                assert panel._prices == {"AAPL": 180.0, "MSFT": 300.0}
                # Total value label reflects snapshot
                total_label = pilot.app.query_one("#portfolio-total", Label)
                # AAPL 10*180 + MSFT 20*300 = 1800 + 6000 = 7800
                assert "7,800" in str(total_label.render())

    @pytest.mark.asyncio
    async def test_portfolio_refresh_fallback_on_missing_prices(self) -> None:
        holdings = [
            Holding(ticker="AAPL", shares=10, avg_cost=150.0),
            Holding(ticker="MSFT", shares=20, avg_cost=300.0),
        ]
        cfg = _make_config(holdings=holdings)
        # Only AAPL available; MSFT fetch will fail
        fake = _FakeRegistry({"AAPL": 180.0})

        with patch("qracer.dashboard.panels.load_config", return_value=cfg):
            async with _SinglePanelApp(PortfolioPanel, data_registry=fake).run_test(
                size=(100, 30)
            ) as pilot:
                panel = pilot.app.query_one(PortfolioPanel)
                await panel.refresh_data()
                # Without complete price set, falls back to cost-basis view
                total_label = pilot.app.query_one("#portfolio-total", Label)
                assert "Cost basis" in str(total_label.render())


# ---------------------------------------------------------------------------
# WatchlistPanel
# ---------------------------------------------------------------------------


class TestWatchlistPanel:
    @pytest.mark.asyncio
    async def test_watchlist_empty(self, tmp_path: Path) -> None:
        with patch("qracer.dashboard.panels._watchlist_path", return_value=tmp_path / "wl.json"):
            async with _SinglePanelApp(WatchlistPanel).run_test(size=(80, 24)) as pilot:
                hint = pilot.app.query_one("#watchlist-hint", Label)
                assert "empty" in str(hint.render()).lower()

    @pytest.mark.asyncio
    async def test_watchlist_with_tickers(self, tmp_path: Path) -> None:
        wl_path = tmp_path / "watchlist.json"
        wl_path.write_text(json.dumps(["AAPL", "TSLA", "NVDA"]))
        with patch("qracer.dashboard.panels._watchlist_path", return_value=wl_path):
            async with _SinglePanelApp(WatchlistPanel).run_test(size=(80, 24)) as pilot:
                table = pilot.app.query_one("#watchlist-table", DataTable)
                assert table.row_count == 3

    @pytest.mark.asyncio
    async def test_watchlist_refresh_updates_prices(self, tmp_path: Path) -> None:
        wl_path = tmp_path / "watchlist.json"
        wl_path.write_text(json.dumps(["AAPL", "TSLA"]))
        fake = _FakeRegistry({"AAPL": 180.0, "TSLA": 250.0})

        with patch("qracer.dashboard.panels._watchlist_path", return_value=wl_path):
            async with _SinglePanelApp(WatchlistPanel, data_registry=fake).run_test(
                size=(80, 24)
            ) as pilot:
                panel = pilot.app.query_one(WatchlistPanel)
                await panel.refresh_data()
                assert panel._prices == {"AAPL": 180.0, "TSLA": 250.0}
                assert set(fake.calls) == {"AAPL", "TSLA"}


# ---------------------------------------------------------------------------
# AlertsPanel
# ---------------------------------------------------------------------------


class TestAlertsPanel:
    @pytest.mark.asyncio
    async def test_alerts_no_store_shows_hint(self, tmp_path: Path) -> None:
        with patch("qracer.dashboard.panels._user_dir", return_value=tmp_path):
            async with _SinglePanelApp(AlertsPanel).run_test(size=(80, 24)) as pilot:
                hint = pilot.app.query_one("#alerts-hint", Label)
                assert "No alerts" in str(hint.render())
                table = pilot.app.query_one("#alerts-table", DataTable)
                assert table.row_count == 0

    @pytest.mark.asyncio
    async def test_alerts_with_active_entries(self, tmp_path: Path) -> None:
        store = AlertStore(tmp_path / "alerts.json")
        store.create("AAPL", AlertCondition.ABOVE, 200.0)
        store.create("TSLA", AlertCondition.BELOW, 180.0)

        async with _SinglePanelApp(AlertsPanel, alert_store=store).run_test(size=(80, 24)) as pilot:
            table = pilot.app.query_one("#alerts-table", DataTable)
            assert table.row_count == 2

    @pytest.mark.asyncio
    async def test_alerts_hot_reload(self, tmp_path: Path) -> None:
        store = AlertStore(tmp_path / "alerts.json")
        async with _SinglePanelApp(AlertsPanel, alert_store=store).run_test(size=(80, 24)) as pilot:
            panel = pilot.app.query_one(AlertsPanel)
            table = pilot.app.query_one("#alerts-table", DataTable)
            assert table.row_count == 0

            # Another "process" adds an alert via a second store instance
            other = AlertStore(tmp_path / "alerts.json")
            other.create("AAPL", AlertCondition.ABOVE, 200.0)

            panel.refresh_data()
            assert table.row_count == 1


# ---------------------------------------------------------------------------
# TasksPanel
# ---------------------------------------------------------------------------


class TestTasksPanel:
    @pytest.mark.asyncio
    async def test_tasks_no_store_shows_hint(self, tmp_path: Path) -> None:
        with patch("qracer.dashboard.panels._user_dir", return_value=tmp_path):
            async with _SinglePanelApp(TasksPanel).run_test(size=(80, 24)) as pilot:
                hint = pilot.app.query_one("#tasks-hint", Label)
                assert "No scheduled tasks" in str(hint.render())
                table = pilot.app.query_one("#tasks-table", DataTable)
                assert table.row_count == 0

    @pytest.mark.asyncio
    async def test_tasks_with_active_entries(self, tmp_path: Path) -> None:
        store = TaskStore(tmp_path / "tasks.json")
        store.create(TaskActionType.ANALYZE, {"ticker": "AAPL"}, "every 1h")
        store.create(TaskActionType.NEWS_SCAN, {"ticker": "TSLA"}, "every 30m")

        async with _SinglePanelApp(TasksPanel, task_store=store).run_test(size=(100, 24)) as pilot:
            table = pilot.app.query_one("#tasks-table", DataTable)
            assert table.row_count == 2

    @pytest.mark.asyncio
    async def test_tasks_refresh_hot_reloads(self, tmp_path: Path) -> None:
        store = TaskStore(tmp_path / "tasks.json")
        async with _SinglePanelApp(TasksPanel, task_store=store).run_test(size=(100, 24)) as pilot:
            panel = pilot.app.query_one(TasksPanel)
            table = pilot.app.query_one("#tasks-table", DataTable)
            assert table.row_count == 0

            other = TaskStore(tmp_path / "tasks.json")
            other.create(TaskActionType.ANALYZE, {"ticker": "MSFT"}, "every 1h")

            panel.refresh_data()
            assert table.row_count == 1


# ---------------------------------------------------------------------------
# NewChatPanel
# ---------------------------------------------------------------------------


class TestNewChatPanel:
    @pytest.mark.asyncio
    async def test_new_chat_renders(self) -> None:
        async with _SinglePanelApp(NewChatPanel).run_test(size=(80, 24)) as pilot:
            labels = pilot.app.query(Label)
            assert any("New Chat" in str(lbl.render()) for lbl in labels)


# ---------------------------------------------------------------------------
# HistoryPanel
# ---------------------------------------------------------------------------


class TestHistoryPanel:
    @pytest.mark.asyncio
    async def test_history_no_sessions(self, tmp_path: Path) -> None:
        with patch("qracer.dashboard.panels._user_dir", return_value=tmp_path):
            async with _SinglePanelApp(HistoryPanel).run_test(size=(80, 24)) as pilot:
                table = pilot.app.query_one("#history-table", DataTable)
                assert table.row_count == 0

    @pytest.mark.asyncio
    async def test_history_with_sessions(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        (sessions_dir / "abc123.jsonl").write_text('{"turn": 1}\n{"turn": 2}\n')
        (sessions_dir / "def456.jsonl").write_text('{"turn": 1}\n')
        with patch("qracer.dashboard.panels._user_dir", return_value=tmp_path):
            async with _SinglePanelApp(HistoryPanel).run_test(size=(80, 24)) as pilot:
                table = pilot.app.query_one("#history-table", DataTable)
                assert table.row_count == 2


# ---------------------------------------------------------------------------
# GeneralSettingsPanel
# ---------------------------------------------------------------------------


class TestGeneralSettingsPanel:
    @pytest.mark.asyncio
    async def test_settings_display(self) -> None:
        cfg = _make_config()
        with patch("qracer.dashboard.panels.load_config", return_value=cfg):
            async with _SinglePanelApp(GeneralSettingsPanel).run_test(size=(80, 24)) as pilot:
                table = pilot.app.query_one("#general-settings-table", DataTable)
                assert table.row_count == 5  # language, mode, provider, model, currency


# ---------------------------------------------------------------------------
# ProvidersSettingsPanel
# ---------------------------------------------------------------------------


class TestProvidersSettingsPanel:
    @pytest.mark.asyncio
    async def test_providers_empty(self) -> None:
        cfg = _make_config()
        with patch("qracer.dashboard.panels.load_config", return_value=cfg):
            async with _SinglePanelApp(ProvidersSettingsPanel).run_test(size=(80, 24)) as pilot:
                table = pilot.app.query_one("#providers-table", DataTable)
                assert table.row_count == 0

    @pytest.mark.asyncio
    async def test_providers_with_entries(self) -> None:
        providers = {
            "yfinance": ProviderConfig(enabled=True, kind="data", tier="hot", priority=10),
            "claude": ProviderConfig(
                enabled=True, kind="llm", api_key_env="ANTHROPIC_API_KEY", priority=100
            ),
        }
        cfg = _make_config(providers=providers)
        with patch("qracer.dashboard.panels.load_config", return_value=cfg):
            async with _SinglePanelApp(ProvidersSettingsPanel).run_test(size=(80, 24)) as pilot:
                table = pilot.app.query_one("#providers-table", DataTable)
                assert table.row_count == 2


# ---------------------------------------------------------------------------
# NotificationsSettingsPanel
# ---------------------------------------------------------------------------


class TestNotificationsSettingsPanel:
    @pytest.mark.asyncio
    async def test_notifications_renders(self) -> None:
        async with _SinglePanelApp(NotificationsSettingsPanel).run_test(size=(80, 24)) as pilot:
            labels = pilot.app.query(Label)
            assert any("Notifications" in str(lbl.render()) for lbl in labels)
