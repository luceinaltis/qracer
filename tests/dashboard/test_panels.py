"""Tests for dashboard panel widgets using Textual's async testing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Label

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
    WatchlistPanel,
    _format_change,
    _watchlist_path,
)


def _make_config(
    holdings: list[Holding] | None = None,
    providers: dict[str, ProviderConfig] | None = None,
) -> QracerConfig:
    return QracerConfig(
        portfolio=PortfolioConfig(holdings=holdings or []),
        providers=ProvidersConfig(providers=providers or {}),
    )


class _SinglePanelApp(App):
    """Test harness that mounts a single panel."""

    def __init__(self, panel_cls: type, panel_id: str = "test-panel") -> None:
        super().__init__()
        self._panel_cls = panel_cls
        self._panel_id = panel_id

    def compose(self) -> ComposeResult:
        yield self._panel_cls(id=self._panel_id)


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


# ---------------------------------------------------------------------------
# OverviewPanel
# ---------------------------------------------------------------------------


class TestOverviewPanel:
    @pytest.mark.asyncio
    async def test_overview_empty_portfolio(self, tmp_path: Path) -> None:
        cfg = _make_config()
        with patch("qracer.dashboard.panels.load_config", return_value=cfg):
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
        with patch("qracer.dashboard.panels.load_config", return_value=cfg):
            async with _SinglePanelApp(OverviewPanel).run_test(size=(80, 24)) as pilot:
                table = pilot.app.query_one("#overview-portfolio-table", DataTable)
                assert table.row_count == 2

    @pytest.mark.asyncio
    async def test_overview_market_table(self) -> None:
        cfg = _make_config()
        with patch("qracer.dashboard.panels.load_config", return_value=cfg):
            async with _SinglePanelApp(OverviewPanel).run_test(size=(80, 24)) as pilot:
                table = pilot.app.query_one("#overview-market-table", DataTable)
                assert table.row_count == 3  # S&P 500, NASDAQ, VIX


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


# ---------------------------------------------------------------------------
# AlertsPanel
# ---------------------------------------------------------------------------


class TestAlertsPanel:
    @pytest.mark.asyncio
    async def test_alerts_renders(self) -> None:
        async with _SinglePanelApp(AlertsPanel).run_test(size=(80, 24)) as pilot:
            labels = pilot.app.query(Label)
            assert any("Alerts" in str(lbl.render()) for lbl in labels)


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
