"""Tests for the correlation/beta computation module."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import AsyncMock

import pytest

from qracer.data.providers import OHLCV
from qracer.risk.correlation import (
    CorrelationEngine,
    CorrelationResult,
    _correlation,
    _covariance,
    _daily_returns,
    _mean,
    _variance,
    compute_beta,
    compute_correlation_result,
    correlation_adjustment,
)
from qracer.risk.models import HoldingSnapshot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bars(closes: list[float], start: date | None = None) -> list[OHLCV]:
    """Create OHLCV bars from a list of closing prices."""
    base = start or date(2026, 1, 1)
    return [
        OHLCV(
            date=base + timedelta(days=i),
            open=c,
            high=c * 1.01,
            low=c * 0.99,
            close=c,
            volume=1_000_000,
        )
        for i, c in enumerate(closes)
    ]


def _make_holding(ticker: str, weight_pct: float) -> HoldingSnapshot:
    return HoldingSnapshot(
        ticker=ticker,
        shares=100,
        avg_cost=100.0,
        current_price=100.0,
        market_value=10_000.0,
        weight_pct=weight_pct,
        unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0,
    )


# ---------------------------------------------------------------------------
# Pure math functions
# ---------------------------------------------------------------------------


class TestDailyReturns:
    def test_basic_returns(self) -> None:
        bars = _make_bars([100.0, 110.0, 105.0])
        returns = _daily_returns(bars)
        assert len(returns) == 2
        assert returns[0] == pytest.approx(0.1, abs=1e-9)
        assert returns[1] == pytest.approx(-0.04545, abs=1e-4)

    def test_single_bar_returns_empty(self) -> None:
        bars = _make_bars([100.0])
        assert _daily_returns(bars) == []

    def test_empty_bars_returns_empty(self) -> None:
        assert _daily_returns([]) == []


class TestStatsFunctions:
    def test_mean(self) -> None:
        assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
        assert _mean([]) == 0.0

    def test_variance(self) -> None:
        # var([1,2,3]) = 1.0 (sample variance)
        assert _variance([1.0, 2.0, 3.0]) == pytest.approx(1.0)
        assert _variance([5.0]) == 0.0
        assert _variance([]) == 0.0

    def test_covariance_identical_series(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0]
        assert _covariance(xs, xs) == pytest.approx(_variance(xs))

    def test_covariance_short_series(self) -> None:
        assert _covariance([1.0], [2.0]) == 0.0

    def test_correlation_identical_series(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _correlation(xs, xs) == pytest.approx(1.0)

    def test_correlation_opposite_series(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert _correlation(xs, ys) == pytest.approx(-1.0)

    def test_correlation_zero_variance(self) -> None:
        xs = [1.0, 1.0, 1.0]
        ys = [2.0, 3.0, 4.0]
        assert _correlation(xs, ys) == 0.0


class TestComputeBeta:
    def test_beta_identical_to_benchmark(self) -> None:
        returns = [0.01, -0.02, 0.015, -0.005, 0.02]
        assert compute_beta(returns, returns) == pytest.approx(1.0)

    def test_beta_double_benchmark(self) -> None:
        bench = [0.01, -0.02, 0.015, -0.005, 0.02]
        stock = [r * 2 for r in bench]
        assert compute_beta(stock, bench) == pytest.approx(2.0, abs=1e-6)

    def test_beta_insufficient_data(self) -> None:
        assert compute_beta([0.01], [0.02]) == 1.0

    def test_beta_zero_variance_benchmark(self) -> None:
        assert compute_beta([0.01, 0.02], [0.0, 0.0]) == 1.0


# ---------------------------------------------------------------------------
# compute_correlation_result
# ---------------------------------------------------------------------------


class TestComputeCorrelationResult:
    def test_single_holding(self) -> None:
        holdings = [_make_holding("AAPL", 100.0)]
        closes = [100.0, 101.0, 102.0, 101.5, 103.0]
        ohlcv = {"AAPL": _make_bars(closes)}
        bench = _make_bars([100.0, 100.5, 101.0, 100.8, 101.5])

        result = compute_correlation_result(holdings, ohlcv, bench)

        assert isinstance(result, CorrelationResult)
        assert "AAPL" in result.betas
        assert result.correlation_avg == 0.0  # only one holding, no pairs
        assert result.correlation_matrix["AAPL"]["AAPL"] == 1.0

    def test_two_correlated_holdings(self) -> None:
        holdings = [_make_holding("A", 50.0), _make_holding("B", 50.0)]
        # A and B move together
        closes_a = [100.0, 102.0, 104.0, 103.0, 105.0]
        closes_b = [50.0, 51.0, 52.0, 51.5, 52.5]
        ohlcv = {"A": _make_bars(closes_a), "B": _make_bars(closes_b)}
        bench = _make_bars([200.0, 201.0, 202.0, 201.5, 203.0])

        result = compute_correlation_result(holdings, ohlcv, bench)

        assert result.correlation_avg > 0.5
        assert result.correlation_matrix["A"]["B"] == result.correlation_matrix["B"]["A"]

    def test_empty_holdings(self) -> None:
        result = compute_correlation_result([], {}, _make_bars([100.0, 101.0]))
        assert result.portfolio_beta == 1.0
        assert result.correlation_avg == 0.0

    def test_portfolio_beta_weighted(self) -> None:
        # Two holdings: A has beta ~2, B has beta ~0.5
        holdings = [_make_holding("A", 60.0), _make_holding("B", 40.0)]
        bench_closes = [100.0, 101.0, 100.5, 101.5, 102.0, 101.0, 102.5]
        bench = _make_bars(bench_closes)
        bench_returns = _daily_returns(bench)

        # A moves 2x benchmark
        a_closes = [100.0]
        for r in bench_returns:
            a_closes.append(a_closes[-1] * (1 + r * 2))

        # B moves 0.5x benchmark
        b_closes = [100.0]
        for r in bench_returns:
            b_closes.append(b_closes[-1] * (1 + r * 0.5))

        ohlcv = {"A": _make_bars(a_closes), "B": _make_bars(b_closes)}
        result = compute_correlation_result(holdings, ohlcv, bench)

        assert result.betas["A"] == pytest.approx(2.0, abs=0.1)
        assert result.betas["B"] == pytest.approx(0.5, abs=0.1)
        # Weighted: 0.6 * 2.0 + 0.4 * 0.5 = 1.4
        assert result.portfolio_beta == pytest.approx(1.4, abs=0.15)


# ---------------------------------------------------------------------------
# CorrelationEngine
# ---------------------------------------------------------------------------


class TestCorrelationEngine:
    @pytest.mark.asyncio
    async def test_compute_returns_result(self) -> None:
        provider = AsyncMock()
        closes = [100.0, 101.0, 102.0, 101.5, 103.0]
        provider.get_ohlcv = AsyncMock(return_value=_make_bars(closes))

        engine = CorrelationEngine(price_provider=provider)
        holdings = [_make_holding("AAPL", 100.0)]

        result = await engine.compute(holdings, as_of=date(2026, 4, 1))

        assert result is not None
        assert result.portfolio_beta is not None
        assert "AAPL" in result.betas

    @pytest.mark.asyncio
    async def test_compute_caches_result(self) -> None:
        provider = AsyncMock()
        closes = [100.0, 101.0, 102.0]
        provider.get_ohlcv = AsyncMock(return_value=_make_bars(closes))

        engine = CorrelationEngine(price_provider=provider)
        holdings = [_make_holding("AAPL", 100.0)]
        as_of = date(2026, 4, 1)

        r1 = await engine.compute(holdings, as_of=as_of)
        r2 = await engine.compute(holdings, as_of=as_of)

        assert r1 is r2  # same cached object
        # get_ohlcv called only for first compute (AAPL + SPY = 2 calls)
        assert provider.get_ohlcv.call_count == 2

    @pytest.mark.asyncio
    async def test_compute_no_provider_returns_none(self) -> None:
        engine = CorrelationEngine(price_provider=None)
        holdings = [_make_holding("AAPL", 100.0)]
        assert await engine.compute(holdings) is None

    @pytest.mark.asyncio
    async def test_compute_empty_holdings_returns_none(self) -> None:
        engine = CorrelationEngine(price_provider=AsyncMock())
        assert await engine.compute([]) is None

    @pytest.mark.asyncio
    async def test_compute_fetch_failure_returns_none(self) -> None:
        provider = AsyncMock()
        provider.get_ohlcv = AsyncMock(side_effect=RuntimeError("network error"))

        engine = CorrelationEngine(price_provider=provider)
        holdings = [_make_holding("AAPL", 100.0)]
        result = await engine.compute(holdings, as_of=date(2026, 4, 1))
        assert result is None

    def test_get_cached_miss(self) -> None:
        engine = CorrelationEngine()
        assert engine.get_cached(["AAPL"], date(2026, 4, 1)) is None

    def test_clear_cache(self) -> None:
        engine = CorrelationEngine()
        engine._cache["test"] = CorrelationResult(
            portfolio_beta=1.0,
            correlation_avg=0.5,
            betas={},
            correlation_matrix={},
        )
        engine.clear_cache()
        assert len(engine._cache) == 0


# ---------------------------------------------------------------------------
# correlation_adjustment
# ---------------------------------------------------------------------------


class TestCorrelationAdjustment:
    def test_no_corr_result_returns_1(self) -> None:
        holdings = [_make_holding("AAPL", 50.0)]
        assert correlation_adjustment("MSFT", holdings, None) == 1.0

    def test_ticker_not_in_matrix_returns_1(self) -> None:
        result = CorrelationResult(
            portfolio_beta=1.0,
            correlation_avg=0.5,
            betas={"AAPL": 1.1},
            correlation_matrix={"AAPL": {"AAPL": 1.0}},
        )
        holdings = [_make_holding("AAPL", 50.0)]
        assert correlation_adjustment("MSFT", holdings, result) == 1.0

    def test_low_correlation_no_reduction(self) -> None:
        result = CorrelationResult(
            portfolio_beta=1.0,
            correlation_avg=0.3,
            betas={"AAPL": 1.1, "MSFT": 1.0},
            correlation_matrix={
                "AAPL": {"AAPL": 1.0, "MSFT": 0.3},
                "MSFT": {"MSFT": 1.0, "AAPL": 0.3},
            },
        )
        holdings = [_make_holding("AAPL", 50.0)]
        adj = correlation_adjustment("MSFT", holdings, result)
        assert adj == 1.0  # below threshold

    def test_high_correlation_reduces_allocation(self) -> None:
        result = CorrelationResult(
            portfolio_beta=1.0,
            correlation_avg=0.9,
            betas={"AAPL": 1.1, "MSFT": 1.0},
            correlation_matrix={
                "AAPL": {"AAPL": 1.0, "MSFT": 0.95},
                "MSFT": {"MSFT": 1.0, "AAPL": 0.95},
            },
        )
        holdings = [_make_holding("AAPL", 100.0)]
        adj = correlation_adjustment("MSFT", holdings, result)
        assert adj < 1.0
        assert adj >= 0.5

    def test_empty_holdings_returns_1(self) -> None:
        result = CorrelationResult(
            portfolio_beta=1.0,
            correlation_avg=0.0,
            betas={},
            correlation_matrix={"MSFT": {"MSFT": 1.0}},
        )
        assert correlation_adjustment("MSFT", [], result) == 1.0


# ---------------------------------------------------------------------------
# RiskCalculator integration with correlation
# ---------------------------------------------------------------------------


class TestRiskCalculatorCorrelationIntegration:
    @pytest.mark.asyncio
    async def test_build_exposure_async_with_engine(self) -> None:
        from qracer.config.models import Holding, PortfolioConfig, PortfolioLimits
        from qracer.risk.calculator import RiskCalculator

        provider = AsyncMock()
        closes = [100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5]
        provider.get_ohlcv = AsyncMock(return_value=_make_bars(closes))

        engine = CorrelationEngine(price_provider=provider)
        config = PortfolioConfig(
            currency="USD",
            holdings=[
                Holding(ticker="AAPL", shares=100, avg_cost=150.0),
                Holding(ticker="JPM", shares=200, avg_cost=140.0),
            ],
            limits=PortfolioLimits(),
        )
        calc = RiskCalculator(config, correlation_engine=engine)
        snapshot = calc.build_snapshot({"AAPL": 180.0, "JPM": 160.0})
        exposure = await calc.build_exposure_async(snapshot)

        assert exposure.portfolio_beta is not None
        assert exposure.correlation_avg is not None

    @pytest.mark.asyncio
    async def test_build_exposure_async_without_engine(self) -> None:
        from qracer.config.models import Holding, PortfolioConfig, PortfolioLimits
        from qracer.risk.calculator import RiskCalculator

        config = PortfolioConfig(
            currency="USD",
            holdings=[Holding(ticker="AAPL", shares=100, avg_cost=150.0)],
            limits=PortfolioLimits(),
        )
        calc = RiskCalculator(config)
        snapshot = calc.build_snapshot({"AAPL": 180.0})
        exposure = await calc.build_exposure_async(snapshot)

        # Falls back to sync build_exposure, beta/corr remain None
        assert exposure.portfolio_beta is None
        assert exposure.correlation_avg is None

    def test_size_position_with_correlation(self) -> None:
        from qracer.config.models import Holding, PortfolioConfig, PortfolioLimits
        from qracer.risk.calculator import RiskCalculator

        config = PortfolioConfig(
            currency="USD",
            holdings=[Holding(ticker="AAPL", shares=100, avg_cost=150.0)],
            limits=PortfolioLimits(max_single_position_pct=15.0, max_sector_pct=100.0),
        )
        calc = RiskCalculator(config)
        snapshot = calc.build_snapshot({"AAPL": 180.0})

        # XOM is Energy sector — no sector constraint with AAPL (Technology).
        # Without correlation: conviction 8 -> 3%
        pct_no_corr = calc.size_position("XOM", 8, snapshot)

        # With high correlation to AAPL (hypothetical cross-sector correlation)
        corr_result = CorrelationResult(
            portfolio_beta=1.0,
            correlation_avg=0.9,
            betas={"AAPL": 1.1, "XOM": 1.0},
            correlation_matrix={
                "AAPL": {"AAPL": 1.0, "XOM": 0.95},
                "XOM": {"XOM": 1.0, "AAPL": 0.95},
            },
        )
        pct_with_corr = calc.size_position("XOM", 8, snapshot, corr_result=corr_result)

        assert pct_with_corr < pct_no_corr

    def test_size_position_backward_compatible(self) -> None:
        """Existing tests should still pass without correlation data."""
        from qracer.config.models import Holding, PortfolioConfig, PortfolioLimits
        from qracer.risk.calculator import RiskCalculator

        config = PortfolioConfig(
            currency="USD",
            holdings=[
                Holding(ticker="AAPL", shares=100, avg_cost=150.0),
                Holding(ticker="MSFT", shares=50, avg_cost=300.0),
                Holding(ticker="JPM", shares=200, avg_cost=140.0),
            ],
            limits=PortfolioLimits(
                max_single_position_pct=15.0,
                max_sector_pct=40.0,
                max_drawdown_alert_pct=10.0,
            ),
        )
        calc = RiskCalculator(config)
        snap = calc.build_snapshot({"AAPL": 180.0, "MSFT": 350.0, "JPM": 160.0})
        # XOM conviction 8 -> base 3%, Energy sector with 0% existing -> 3%
        pct = calc.size_position("XOM", 8, snap)
        assert pct == 3.0

    @pytest.mark.asyncio
    async def test_assess_async(self) -> None:
        from qracer.config.models import Holding, PortfolioConfig, PortfolioLimits
        from qracer.risk.calculator import RiskCalculator

        provider = AsyncMock()
        closes = [100.0, 101.0, 102.0, 101.5, 103.0]
        provider.get_ohlcv = AsyncMock(return_value=_make_bars(closes))

        engine = CorrelationEngine(price_provider=provider)
        config = PortfolioConfig(
            currency="USD",
            holdings=[Holding(ticker="AAPL", shares=100, avg_cost=150.0)],
            limits=PortfolioLimits(),
        )
        calc = RiskCalculator(config, correlation_engine=engine)
        result = await calc.assess_async({"AAPL": 180.0})

        assert result.exposure.portfolio_beta is not None
        assert result.snapshot.total_value == 18000.0
