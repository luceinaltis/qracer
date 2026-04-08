"""Backtesting engine for trade thesis validation against historical data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from qracer.data.providers import OHLCV
    from qracer.data.registry import DataRegistry
    from qracer.models.base import TradeThesis


@dataclass(frozen=True)
class Trade:
    """A single simulated trade."""

    outcome: Literal["win", "loss"]
    entry_price: float
    exit_price: float
    entry_date: date
    exit_date: date

    @property
    def return_pct(self) -> float:
        return (self.exit_price - self.entry_price) / self.entry_price * 100

    @property
    def holding_days(self) -> int:
        return (self.exit_date - self.entry_date).days


@dataclass
class BacktestResult:
    """Aggregated results of a backtest run."""

    ticker: str
    period_days: int
    entry_count: int
    target_hit_count: int
    stop_hit_count: int
    win_rate: float
    avg_holding_days: float
    avg_return_pct: float
    max_drawdown_pct: float
    historical_rr: float
    trades: list[Trade] = field(default_factory=list)


class Backtester:
    """Simulates a trade thesis against historical OHLCV data."""

    def __init__(self, data_registry: DataRegistry) -> None:
        self._data = data_registry

    async def run(
        self,
        thesis: TradeThesis,
        lookback_days: int = 180,
    ) -> BacktestResult:
        """Fetch historical bars and simulate the thesis entry/exit logic."""
        from qracer.data.providers import PriceProvider

        end = date.today()
        start = end - timedelta(days=lookback_days)
        bars: list[OHLCV] = await self._data.async_get_with_fallback(
            PriceProvider, "get_ohlcv", thesis.ticker, start, end
        )
        return self._simulate(thesis, bars, lookback_days)

    def _simulate(
        self, thesis: TradeThesis, bars: list[OHLCV], lookback_days: int
    ) -> BacktestResult:
        """Walk through bars, track entries and exits."""
        trades: list[Trade] = []
        in_trade = False
        entry_price = 0.0
        entry_date = bars[0].date if bars else date.today()

        for bar in bars:
            if not in_trade:
                # Check if price enters entry zone
                if thesis.entry_zone[0] <= bar.close <= thesis.entry_zone[1]:
                    in_trade = True
                    entry_price = bar.close
                    entry_date = bar.date
            else:
                # Check exit conditions: target hit
                if bar.high >= thesis.target_price:
                    trades.append(
                        Trade("win", entry_price, thesis.target_price, entry_date, bar.date)
                    )
                    in_trade = False
                # Check exit conditions: stop loss hit
                elif bar.low <= thesis.stop_loss:
                    trades.append(
                        Trade("loss", entry_price, thesis.stop_loss, entry_date, bar.date)
                    )
                    in_trade = False

        return self._compute_result(thesis.ticker, lookback_days, trades)

    def _compute_result(
        self, ticker: str, lookback_days: int, trades: list[Trade]
    ) -> BacktestResult:
        """Aggregate individual trades into a BacktestResult."""
        if not trades:
            return BacktestResult(
                ticker=ticker,
                period_days=lookback_days,
                entry_count=0,
                target_hit_count=0,
                stop_hit_count=0,
                win_rate=0.0,
                avg_holding_days=0.0,
                avg_return_pct=0.0,
                max_drawdown_pct=0.0,
                historical_rr=0.0,
                trades=[],
            )

        wins = [t for t in trades if t.outcome == "win"]
        losses = [t for t in trades if t.outcome == "loss"]
        total = len(trades)

        win_rate = len(wins) / total * 100
        avg_holding = sum(t.holding_days for t in trades) / total
        avg_return = sum(t.return_pct for t in trades) / total

        # Max drawdown: worst single-trade return
        max_drawdown = min(t.return_pct for t in trades)

        # Historical risk/reward: average win / abs(average loss)
        avg_win = sum(t.return_pct for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(t.return_pct for t in losses) / len(losses) if losses else 0.0
        if avg_loss != 0:
            historical_rr = avg_win / abs(avg_loss)
        elif avg_win > 0:
            historical_rr = float("inf")
        else:
            historical_rr = 0.0

        return BacktestResult(
            ticker=ticker,
            period_days=lookback_days,
            entry_count=total,
            target_hit_count=len(wins),
            stop_hit_count=len(losses),
            win_rate=win_rate,
            avg_holding_days=avg_holding,
            avg_return_pct=avg_return,
            max_drawdown_pct=max_drawdown,
            historical_rr=historical_rr,
            trades=trades,
        )


def format_backtest_result(result: BacktestResult, thesis: TradeThesis) -> str:
    """Format a BacktestResult for terminal display."""
    entry_lo, entry_hi = thesis.entry_zone
    lines = [
        f"Backtesting {result.ticker} thesis "
        f"(entry ${entry_lo:.0f}-${entry_hi:.0f}, "
        f"target ${thesis.target_price:.0f}, stop ${thesis.stop_loss:.0f})",
        f"Period: {result.period_days} days",
        "",
    ]

    if result.entry_count == 0:
        lines.append("No simulated trades — price never entered the entry zone.")
        return "\n".join(lines)

    open_trades = result.entry_count - result.target_hit_count - result.stop_hit_count
    lines.append(f"Simulated Trades: {result.entry_count}")
    lines.append(f"  \u2713 Win:  {result.target_hit_count} ({result.win_rate:.0f}%)")
    lines.append(f"  \u2717 Loss: {result.stop_hit_count} ({100 - result.win_rate:.0f}%)")
    if open_trades > 0:
        lines.append(f"  \u25cc Open: {open_trades}")
    lines.append("")
    lines.append(f"Avg Holding: {result.avg_holding_days:.0f} days")
    lines.append(f"Avg Return:  {result.avg_return_pct:+.1f}%")
    lines.append(f"Max Drawdown: {result.max_drawdown_pct:.1f}%")

    thesis_rr = f"{thesis.risk_reward_ratio:.2f}x"
    if result.historical_rr == float("inf"):
        hist_rr = "\u221e"
    else:
        hist_rr = f"{result.historical_rr:.2f}x"
    lines.append(f"Historical R/R: {hist_rr} (thesis: {thesis_rr})")

    return "\n".join(lines)
