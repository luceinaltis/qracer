"""Tests for backtest module — simulation logic and result formatting."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock

import pytest

from qracer.backtest import BacktestResult, Backtester, Trade, format_backtest_result
from qracer.data.providers import OHLCV, PriceProvider
from qracer.data.registry import DataRegistry
from qracer.models.base import TradeThesis


def _make_thesis(
    ticker: str = "AAPL",
    entry_zone: tuple[float, float] = (170.0, 175.0),
    target_price: float = 200.0,
    stop_loss: float = 160.0,
) -> TradeThesis:
    mid = (entry_zone[0] + entry_zone[1]) / 2
    rr = (target_price - mid) / (mid - stop_loss)
    return TradeThesis(
        ticker=ticker,
        entry_zone=entry_zone,
        target_price=target_price,
        stop_loss=stop_loss,
        risk_reward_ratio=rr,
        catalyst="Earnings beat expected",
        catalyst_date="2026-Q2",
        conviction=7,
        summary="Strong upside thesis",
    )


def _bar(d: date, o: float, h: float, l: float, c: float, v: int = 1_000_000) -> OHLCV:
    return OHLCV(date=d, open=o, high=h, low=l, close=c, volume=v)


def _registry_with_bars(bars: list[OHLCV]) -> DataRegistry:
    mock = AsyncMock(spec=PriceProvider)
    mock.get_ohlcv.return_value = bars
    reg = DataRegistry()
    reg.register("mock", mock, [PriceProvider])
    return reg


# ---------------------------------------------------------------------------
# Trade dataclass
# ---------------------------------------------------------------------------


class TestTrade:
    def test_return_pct_win(self) -> None:
        t = Trade("win", 170.0, 200.0, date(2026, 1, 1), date(2026, 1, 20))
        assert abs(t.return_pct - 17.647) < 0.01

    def test_return_pct_loss(self) -> None:
        t = Trade("loss", 170.0, 160.0, date(2026, 1, 1), date(2026, 1, 5))
        assert abs(t.return_pct - (-5.882)) < 0.01

    def test_holding_days(self) -> None:
        t = Trade("win", 170.0, 200.0, date(2026, 1, 1), date(2026, 1, 25))
        assert t.holding_days == 24


# ---------------------------------------------------------------------------
# Backtester._simulate — core logic
# ---------------------------------------------------------------------------


class TestSimulate:
    def test_no_bars_returns_empty(self) -> None:
        bt = Backtester.__new__(Backtester)
        thesis = _make_thesis()
        result = bt._simulate(thesis, [], 180)
        assert result.entry_count == 0
        assert result.win_rate == 0.0

    def test_price_never_enters_zone(self) -> None:
        """If price stays above entry zone, no trades should fire."""
        bt = Backtester.__new__(Backtester)
        thesis = _make_thesis(entry_zone=(170.0, 175.0))
        bars = [
            _bar(date(2026, 1, i), 180.0, 185.0, 178.0, 180.0) for i in range(1, 11)
        ]
        result = bt._simulate(thesis, bars, 180)
        assert result.entry_count == 0

    def test_single_winning_trade(self) -> None:
        """Price enters zone, then hits target."""
        bt = Backtester.__new__(Backtester)
        thesis = _make_thesis(entry_zone=(170.0, 175.0), target_price=200.0, stop_loss=160.0)
        bars = [
            _bar(date(2026, 1, 1), 180.0, 182.0, 170.0, 172.0),  # enter zone
            _bar(date(2026, 1, 2), 175.0, 180.0, 173.0, 178.0),  # holding
            _bar(date(2026, 1, 3), 180.0, 201.0, 179.0, 199.0),  # target hit (high >= 200)
        ]
        result = bt._simulate(thesis, bars, 180)
        assert result.entry_count == 1
        assert result.target_hit_count == 1
        assert result.stop_hit_count == 0
        assert result.win_rate == 100.0

    def test_single_losing_trade(self) -> None:
        """Price enters zone, then hits stop loss."""
        bt = Backtester.__new__(Backtester)
        thesis = _make_thesis(entry_zone=(170.0, 175.0), target_price=200.0, stop_loss=160.0)
        bars = [
            _bar(date(2026, 1, 1), 174.0, 176.0, 170.0, 173.0),  # enter zone
            _bar(date(2026, 1, 2), 170.0, 172.0, 165.0, 167.0),  # holding
            _bar(date(2026, 1, 3), 165.0, 166.0, 159.0, 162.0),  # stop hit (low <= 160)
        ]
        result = bt._simulate(thesis, bars, 180)
        assert result.entry_count == 1
        assert result.target_hit_count == 0
        assert result.stop_hit_count == 1
        assert result.win_rate == 0.0

    def test_multiple_trades(self) -> None:
        """Two completed trades: one win, one loss."""
        bt = Backtester.__new__(Backtester)
        thesis = _make_thesis(entry_zone=(170.0, 175.0), target_price=200.0, stop_loss=160.0)
        bars = [
            # Trade 1: enter and win
            _bar(date(2026, 1, 1), 174.0, 176.0, 170.0, 172.0),  # enter
            _bar(date(2026, 1, 2), 180.0, 201.0, 178.0, 198.0),  # target hit
            # Trade 2: enter and lose
            _bar(date(2026, 1, 3), 174.0, 176.0, 170.0, 171.0),  # re-enter
            _bar(date(2026, 1, 4), 165.0, 168.0, 155.0, 158.0),  # stop hit
        ]
        result = bt._simulate(thesis, bars, 180)
        assert result.entry_count == 2
        assert result.target_hit_count == 1
        assert result.stop_hit_count == 1
        assert abs(result.win_rate - 50.0) < 0.01

    def test_open_trade_not_counted(self) -> None:
        """Entry without exit should not be in the trades list."""
        bt = Backtester.__new__(Backtester)
        thesis = _make_thesis(entry_zone=(170.0, 175.0), target_price=200.0, stop_loss=160.0)
        bars = [
            _bar(date(2026, 1, 1), 174.0, 176.0, 170.0, 172.0),  # enter
            _bar(date(2026, 1, 2), 175.0, 180.0, 173.0, 178.0),  # holding, no exit
        ]
        result = bt._simulate(thesis, bars, 180)
        assert result.entry_count == 0  # no completed trades

    def test_avg_holding_days(self) -> None:
        bt = Backtester.__new__(Backtester)
        thesis = _make_thesis(entry_zone=(170.0, 175.0), target_price=200.0, stop_loss=160.0)
        bars = [
            _bar(date(2026, 1, 1), 174.0, 176.0, 170.0, 172.0),  # enter
            _bar(date(2026, 1, 11), 195.0, 205.0, 193.0, 199.0),  # target hit (10 days)
        ]
        result = bt._simulate(thesis, bars, 180)
        assert result.avg_holding_days == 10.0

    def test_max_drawdown(self) -> None:
        """Max drawdown should be the worst single-trade return."""
        bt = Backtester.__new__(Backtester)
        thesis = _make_thesis(entry_zone=(170.0, 175.0), target_price=200.0, stop_loss=160.0)
        bars = [
            # Win trade
            _bar(date(2026, 1, 1), 174.0, 176.0, 170.0, 172.0),
            _bar(date(2026, 1, 2), 195.0, 205.0, 193.0, 199.0),
            # Loss trade
            _bar(date(2026, 1, 3), 174.0, 176.0, 170.0, 171.0),
            _bar(date(2026, 1, 4), 162.0, 165.0, 155.0, 158.0),
        ]
        result = bt._simulate(thesis, bars, 180)
        # Loss trade: exit at stop_loss=160, entry at 171 => return = (160-171)/171 * 100 ~ -6.43%
        assert result.max_drawdown_pct < 0


# ---------------------------------------------------------------------------
# Backtester.run — async integration
# ---------------------------------------------------------------------------


class TestBacktesterRun:
    async def test_run_fetches_bars_and_simulates(self) -> None:
        bars = [
            _bar(date(2026, 1, 1), 174.0, 176.0, 170.0, 172.0),
            _bar(date(2026, 1, 2), 195.0, 205.0, 193.0, 199.0),
        ]
        registry = _registry_with_bars(bars)
        thesis = _make_thesis()
        bt = Backtester(registry)
        result = await bt.run(thesis, lookback_days=90)
        assert result.ticker == "AAPL"
        assert result.period_days == 90
        assert result.entry_count == 1
        assert result.target_hit_count == 1

    async def test_run_with_no_matching_bars(self) -> None:
        bars = [
            _bar(date(2026, 1, i), 200.0, 210.0, 195.0, 205.0) for i in range(1, 6)
        ]
        registry = _registry_with_bars(bars)
        thesis = _make_thesis(entry_zone=(170.0, 175.0))
        bt = Backtester(registry)
        result = await bt.run(thesis, lookback_days=30)
        assert result.entry_count == 0


# ---------------------------------------------------------------------------
# format_backtest_result
# ---------------------------------------------------------------------------


class TestFormatResult:
    def test_format_with_trades(self) -> None:
        thesis = _make_thesis()
        result = BacktestResult(
            ticker="AAPL",
            period_days=180,
            entry_count=3,
            target_hit_count=2,
            stop_hit_count=1,
            win_rate=66.7,
            avg_holding_days=23.0,
            avg_return_pct=8.2,
            max_drawdown_pct=-3.1,
            historical_rr=3.2,
        )
        text = format_backtest_result(result, thesis)
        assert "AAPL" in text
        assert "entry $170-$175" in text
        assert "target $200" in text
        assert "stop $160" in text
        assert "Simulated Trades: 3" in text
        assert "Win:  2" in text
        assert "Loss: 1" in text
        assert "+8.2%" in text

    def test_format_no_trades(self) -> None:
        thesis = _make_thesis()
        result = BacktestResult(
            ticker="AAPL",
            period_days=180,
            entry_count=0,
            target_hit_count=0,
            stop_hit_count=0,
            win_rate=0.0,
            avg_holding_days=0.0,
            avg_return_pct=0.0,
            max_drawdown_pct=0.0,
            historical_rr=0.0,
        )
        text = format_backtest_result(result, thesis)
        assert "No simulated trades" in text

    def test_format_infinite_rr(self) -> None:
        thesis = _make_thesis()
        result = BacktestResult(
            ticker="AAPL",
            period_days=180,
            entry_count=2,
            target_hit_count=2,
            stop_hit_count=0,
            win_rate=100.0,
            avg_holding_days=15.0,
            avg_return_pct=12.0,
            max_drawdown_pct=0.5,
            historical_rr=float("inf"),
        )
        text = format_backtest_result(result, thesis)
        assert "\u221e" in text
