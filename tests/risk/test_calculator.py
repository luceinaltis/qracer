"""Tests for the risk calculator module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from qracer.config.models import Holding, PortfolioConfig, PortfolioLimits
from qracer.models import TradeThesis
from qracer.risk.calculator import RiskCalculator, SectorResolver, get_sector
from qracer.risk.correlation import CorrelationResult
from qracer.risk.models import PortfolioSnapshot, RebalanceAction

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def portfolio_config() -> PortfolioConfig:
    return PortfolioConfig(
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


@pytest.fixture
def prices() -> dict[str, float]:
    return {"AAPL": 180.0, "MSFT": 350.0, "JPM": 160.0}


@pytest.fixture
def calculator(portfolio_config: PortfolioConfig) -> RiskCalculator:
    return RiskCalculator(portfolio_config)


@pytest.fixture
def snapshot(calculator: RiskCalculator, prices: dict[str, float]) -> PortfolioSnapshot:
    return calculator.build_snapshot(prices)


# ---------------------------------------------------------------------------
# build_snapshot
# ---------------------------------------------------------------------------


class TestBuildSnapshot:
    def test_total_value(self, snapshot: PortfolioSnapshot) -> None:
        # AAPL: 100*180=18000, MSFT: 50*350=17500, JPM: 200*160=32000
        assert snapshot.total_value == 67500.0

    def test_holding_count(self, snapshot: PortfolioSnapshot) -> None:
        assert len(snapshot.holdings) == 3

    def test_market_values(self, snapshot: PortfolioSnapshot) -> None:
        by_ticker = {h.ticker: h for h in snapshot.holdings}
        assert by_ticker["AAPL"].market_value == 18000.0
        assert by_ticker["MSFT"].market_value == 17500.0
        assert by_ticker["JPM"].market_value == 32000.0

    def test_weights_sum_to_100(self, snapshot: PortfolioSnapshot) -> None:
        total_weight = sum(h.weight_pct for h in snapshot.holdings)
        assert abs(total_weight - 100.0) < 0.1

    def test_unrealized_pnl(self, snapshot: PortfolioSnapshot) -> None:
        by_ticker = {h.ticker: h for h in snapshot.holdings}
        # AAPL: (180-150)*100 = 3000
        assert by_ticker["AAPL"].unrealized_pnl == 3000.0
        # AAPL pnl%: 3000 / 15000 * 100 = 20.0
        assert by_ticker["AAPL"].unrealized_pnl_pct == 20.0

    def test_missing_price_skips_holding(self, calculator: RiskCalculator) -> None:
        partial_prices = {"AAPL": 180.0, "JPM": 160.0}
        snap = calculator.build_snapshot(partial_prices)
        tickers = {h.ticker for h in snap.holdings}
        assert "MSFT" not in tickers
        assert len(snap.holdings) == 2

    def test_empty_holdings(self) -> None:
        config = PortfolioConfig(currency="USD", holdings=[])
        calc = RiskCalculator(config)
        snap = calc.build_snapshot({})
        assert snap.total_value == 0.0
        assert snap.holdings == []

    def test_currency_propagated(self, snapshot: PortfolioSnapshot) -> None:
        assert snapshot.currency == "USD"


# ---------------------------------------------------------------------------
# build_exposure
# ---------------------------------------------------------------------------


class TestBuildExposure:
    def test_sector_weights(self, calculator: RiskCalculator, snapshot: PortfolioSnapshot) -> None:
        exposure = calculator.build_exposure(snapshot)
        # Technology: AAPL+MSFT = 18000+17500 = 35500 -> 52.59%
        # Financials: JPM = 32000 -> 47.41%
        assert "Technology" in exposure.sector_weights
        assert "Financials" in exposure.sector_weights
        assert abs(exposure.sector_weights["Technology"] - 52.59) < 0.1
        assert abs(exposure.sector_weights["Financials"] - 47.41) < 0.1

    def test_top_sector(self, calculator: RiskCalculator, snapshot: PortfolioSnapshot) -> None:
        exposure = calculator.build_exposure(snapshot)
        assert exposure.top_sector == "Technology"
        assert abs(exposure.top_sector_pct - 52.59) < 0.1


# ---------------------------------------------------------------------------
# check_limits
# ---------------------------------------------------------------------------


class TestCheckLimits:
    def test_no_breaches_within_limits(self) -> None:
        config = PortfolioConfig(
            currency="USD",
            holdings=[
                Holding(ticker="AAPL", shares=10, avg_cost=150.0),
                Holding(ticker="JPM", shares=10, avg_cost=140.0),
            ],
            limits=PortfolioLimits(max_single_position_pct=60.0, max_sector_pct=60.0),
        )
        calc = RiskCalculator(config)
        snap = calc.build_snapshot({"AAPL": 150.0, "JPM": 150.0})
        exposure = calc.build_exposure(snap)
        breached = calc.check_limits(snap, exposure)
        assert breached == []

    def test_single_position_breach(
        self, calculator: RiskCalculator, snapshot: PortfolioSnapshot
    ) -> None:
        exposure = calculator.build_exposure(snapshot)
        # JPM is ~47.4%, way above 15% limit
        breached = calculator.check_limits(snapshot, exposure)
        position_breaches = [b for b in breached if "weight" in b and "single position" in b]
        assert len(position_breaches) > 0

    def test_sector_breach(self, calculator: RiskCalculator, snapshot: PortfolioSnapshot) -> None:
        exposure = calculator.build_exposure(snapshot)
        # Technology ~52.6% and Financials ~47.4%, both > 40%
        breached = calculator.check_limits(snapshot, exposure)
        sector_breaches = [b for b in breached if "Sector" in b]
        assert len(sector_breaches) > 0


# ---------------------------------------------------------------------------
# size_position
# ---------------------------------------------------------------------------


class TestSizePosition:
    def test_high_conviction(self, calculator: RiskCalculator, snapshot: PortfolioSnapshot) -> None:
        # Conviction 10 -> base 5%
        pct = calculator.size_position("XOM", 10, snapshot)
        assert 0.0 < pct <= 5.0

    def test_moderate_conviction(
        self, calculator: RiskCalculator, snapshot: PortfolioSnapshot
    ) -> None:
        # Conviction 6 -> base 2%
        pct = calculator.size_position("XOM", 6, snapshot)
        assert 0.0 < pct <= 3.0

    def test_low_conviction(self, calculator: RiskCalculator, snapshot: PortfolioSnapshot) -> None:
        # Conviction 2 -> base ~0.67%
        pct = calculator.size_position("XOM", 2, snapshot)
        assert 0.0 < pct <= 1.0

    def test_never_exceeds_max_single_position(self) -> None:
        config = PortfolioConfig(
            currency="USD",
            holdings=[Holding(ticker="AAPL", shares=10, avg_cost=150.0)],
            limits=PortfolioLimits(max_single_position_pct=2.0, max_sector_pct=100.0),
        )
        calc = RiskCalculator(config)
        snap = calc.build_snapshot({"AAPL": 150.0})
        pct = calc.size_position("XOM", 10, snap)
        assert pct <= 2.0

    def test_sector_near_limit_reduces_allocation(self) -> None:
        config = PortfolioConfig(
            currency="USD",
            holdings=[
                Holding(ticker="AAPL", shares=100, avg_cost=150.0),
                Holding(ticker="MSFT", shares=100, avg_cost=300.0),
            ],
            limits=PortfolioLimits(max_single_position_pct=15.0, max_sector_pct=50.0),
        )
        calc = RiskCalculator(config)
        # AAPL: 100*180=18000, MSFT: 100*350=35000, total=53000
        # Technology weight: 100% since all are tech
        snap = calc.build_snapshot({"AAPL": 180.0, "MSFT": 350.0})
        # Adding another tech stock should be constrained: sector already at 100%
        pct = calc.size_position("GOOGL", 10, snap)
        # Sector headroom is max(0, 50 - 100) = 0
        assert pct == 0.0

    def test_conviction_8_base_3_percent(
        self, calculator: RiskCalculator, snapshot: PortfolioSnapshot
    ) -> None:
        pct = calculator.size_position("XOM", 8, snapshot)
        # XOM is Energy sector, no existing energy holdings, so base=3%
        assert pct == 3.0

    def test_conviction_5_base_1_percent(
        self, calculator: RiskCalculator, snapshot: PortfolioSnapshot
    ) -> None:
        pct = calculator.size_position("XOM", 5, snapshot)
        assert pct == 1.0

    def test_conviction_1_base_half_percent(
        self, calculator: RiskCalculator, snapshot: PortfolioSnapshot
    ) -> None:
        pct = calculator.size_position("XOM", 1, snapshot)
        assert pct == 0.5


# ---------------------------------------------------------------------------
# risk_check pipeline tool
# ---------------------------------------------------------------------------


class TestRiskCheckPipeline:
    @pytest.mark.asyncio
    async def test_risk_check_success(self, portfolio_config: PortfolioConfig) -> None:
        from qracer.tools.pipeline import risk_check

        # Mock registry with a PriceProvider.
        mock_provider = AsyncMock()
        mock_provider.get_price = AsyncMock(
            side_effect=lambda t: {"AAPL": 180.0, "MSFT": 350.0, "JPM": 160.0}[t]
        )
        mock_registry = MagicMock()
        mock_registry.get = MagicMock(return_value=mock_provider)

        thesis = TradeThesis(
            ticker="XOM",
            entry_zone=(95.0, 100.0),
            target_price=120.0,
            stop_loss=90.0,
            risk_reward_ratio=2.67,
            catalyst="Oil supply cut",
            catalyst_date="Q2 2026",
            conviction=8,
            summary="Long XOM on supply thesis.",
        )

        result = await risk_check("XOM", thesis, mock_registry, portfolio_config)

        assert result.success is True
        assert result.tool == "risk_check"
        assert result.data["ticker"] == "XOM"
        assert result.data["conviction"] == 8
        assert result.data["allocation_pct"] > 0
        assert "sized_recommendation" in result.data

    @pytest.mark.asyncio
    async def test_risk_check_with_empty_portfolio(self) -> None:
        from qracer.tools.pipeline import risk_check

        config = PortfolioConfig(currency="USD", holdings=[])
        mock_registry = MagicMock()
        # No prices needed for empty portfolio.
        mock_registry.get = MagicMock(return_value=AsyncMock())

        thesis = TradeThesis(
            ticker="AAPL",
            entry_zone=(170.0, 175.0),
            target_price=200.0,
            stop_loss=160.0,
            risk_reward_ratio=2.86,
            catalyst="iPhone launch",
            catalyst_date=None,
            conviction=7,
            summary="Long AAPL.",
        )

        result = await risk_check("AAPL", thesis, mock_registry, config)
        assert result.success is True
        assert result.data["portfolio_value"] == 0.0


# ---------------------------------------------------------------------------
# get_sector helper
# ---------------------------------------------------------------------------


class TestGetSector:
    def test_known_ticker(self) -> None:
        assert get_sector("AAPL") == "Technology"
        assert get_sector("JPM") == "Financials"
        assert get_sector("XOM") == "Energy"

    def test_unknown_ticker(self) -> None:
        assert get_sector("ZZZZ") == "Unknown"

    def test_case_insensitive(self) -> None:
        assert get_sector("aapl") == "Technology"


# ---------------------------------------------------------------------------
# Drawdown tracking
# ---------------------------------------------------------------------------


class TestDrawdownTracking:
    def test_update_peak(self, calculator: RiskCalculator) -> None:
        assert calculator.peak_value == 0.0
        calculator.update_peak(100_000.0)
        assert calculator.peak_value == 100_000.0
        calculator.update_peak(90_000.0)  # lower — should not change
        assert calculator.peak_value == 100_000.0
        calculator.update_peak(110_000.0)  # new high
        assert calculator.peak_value == 110_000.0

    def test_compute_drawdown_at_peak(self, calculator: RiskCalculator) -> None:
        calculator.update_peak(100_000.0)
        assert calculator.compute_drawdown(100_000.0) == 0.0

    def test_compute_drawdown_below_peak(self, calculator: RiskCalculator) -> None:
        calculator.update_peak(100_000.0)
        dd = calculator.compute_drawdown(90_000.0)
        assert dd == pytest.approx(10.0)

    def test_compute_drawdown_above_peak(self, calculator: RiskCalculator) -> None:
        calculator.update_peak(100_000.0)
        assert calculator.compute_drawdown(110_000.0) == 0.0

    def test_compute_drawdown_zero_peak(self, calculator: RiskCalculator) -> None:
        assert calculator.compute_drawdown(50_000.0) == 0.0

    def test_assess_drawdown_no_alert(self, portfolio_config: PortfolioConfig) -> None:
        """Drawdown under threshold should not trigger alert."""
        # AAPL=100*180=18000, MSFT=50*350=17500, JPM=200*160=32000 → total=67500
        calc = RiskCalculator(portfolio_config, peak_value=70_000.0)
        prices = {"AAPL": 180.0, "MSFT": 350.0, "JPM": 160.0}
        result = calc.assess(prices)
        # Drawdown is (70000 - 67500) / 70000 ≈ 3.6% < 10% threshold
        assert result.max_drawdown_alert is False
        assert result.current_drawdown_pct < 10.0
        assert result.peak_value == 70_000.0

    def test_assess_drawdown_triggers_alert(self, portfolio_config: PortfolioConfig) -> None:
        """Drawdown exceeding threshold should trigger alert."""
        # total ≈ 67500, peak=100000 → drawdown ≈ 32.5% > 10%
        calc = RiskCalculator(portfolio_config, peak_value=100_000.0)
        prices = {"AAPL": 180.0, "MSFT": 350.0, "JPM": 160.0}
        result = calc.assess(prices)
        assert result.max_drawdown_alert is True
        assert result.current_drawdown_pct > 10.0
        assert result.peak_value == 100_000.0

    def test_assess_updates_peak_on_new_high(self, portfolio_config: PortfolioConfig) -> None:
        """If current value exceeds peak, peak should update."""
        # total ≈ 67500, peak=50000 → new high
        calc = RiskCalculator(portfolio_config, peak_value=50_000.0)
        prices = {"AAPL": 180.0, "MSFT": 350.0, "JPM": 160.0}
        result = calc.assess(prices)
        assert result.peak_value == result.snapshot.total_value
        assert result.current_drawdown_pct == 0.0
        assert result.max_drawdown_alert is False


# ---------------------------------------------------------------------------
# SectorResolver
# ---------------------------------------------------------------------------


class TestSectorResolver:
    def test_hardcoded_lookup(self) -> None:
        resolver = SectorResolver()
        assert resolver.get_sector("AAPL") == "Technology"
        assert resolver.get_sector("JPM") == "Financials"

    def test_unknown_ticker_returns_unknown(self) -> None:
        resolver = SectorResolver()
        assert resolver.get_sector("ZZZZ") == "Unknown"

    def test_case_insensitive(self) -> None:
        resolver = SectorResolver()
        assert resolver.get_sector("aapl") == "Technology"

    def test_cache_hit(self) -> None:
        resolver = SectorResolver()
        resolver.get_sector("AAPL")
        assert "AAPL" in resolver._cache
        assert resolver.get_sector("AAPL") == "Technology"

    async def test_async_fallback_to_hardcoded(self) -> None:
        resolver = SectorResolver()
        sector = await resolver.get_sector_async("AAPL")
        assert sector == "Technology"

    async def test_async_dynamic_lookup(self) -> None:
        from qracer.data.providers import FundamentalData

        mock_registry = MagicMock()

        async def _fake_fallback(*args, **kwargs):
            return FundamentalData(ticker="ZZZZ", sector="Industrials")

        mock_registry.async_get_with_fallback = _fake_fallback

        resolver = SectorResolver(data_registry=mock_registry)
        sector = await resolver.get_sector_async("ZZZZ")
        assert sector == "Industrials"
        assert resolver._cache["ZZZZ"] == "Industrials"

    async def test_async_no_sector_falls_back(self) -> None:
        from qracer.data.providers import FundamentalData

        mock_registry = MagicMock()

        async def _fake_fallback(*args, **kwargs):
            return FundamentalData(ticker="AAPL", sector=None)

        mock_registry.async_get_with_fallback = _fake_fallback

        resolver = SectorResolver(data_registry=mock_registry)
        sector = await resolver.get_sector_async("AAPL")
        assert sector == "Technology"

    async def test_async_failure_falls_back(self) -> None:
        mock_registry = MagicMock()

        async def _failing(*args, **kwargs):
            raise RuntimeError("down")

        mock_registry.async_get_with_fallback = _failing

        resolver = SectorResolver(data_registry=mock_registry)
        sector = await resolver.get_sector_async("AAPL")
        assert sector == "Technology"

    def test_calculator_uses_resolver(self, portfolio_config: PortfolioConfig) -> None:
        resolver = SectorResolver()
        calc = RiskCalculator(portfolio_config, sector_resolver=resolver)
        prices = {"AAPL": 180.0, "MSFT": 350.0, "JPM": 160.0}
        exposure = calc.build_exposure(calc.build_snapshot(prices))
        assert "Technology" in exposure.sector_weights
        assert "Financials" in exposure.sector_weights


# ---------------------------------------------------------------------------
# suggest_rebalance
# ---------------------------------------------------------------------------


class TestSuggestRebalance:
    def test_no_breaches_returns_empty(self) -> None:
        """When all limits are satisfied, no rebalancing is suggested."""
        config = PortfolioConfig(
            currency="USD",
            holdings=[
                Holding(ticker="AAPL", shares=10, avg_cost=150.0),
                Holding(ticker="JPM", shares=10, avg_cost=140.0),
            ],
            limits=PortfolioLimits(max_single_position_pct=60.0, max_sector_pct=60.0),
        )
        calc = RiskCalculator(config)
        snap = calc.build_snapshot({"AAPL": 150.0, "JPM": 150.0})
        exposure = calc.build_exposure(snap)
        actions = calc.suggest_rebalance(snap, exposure)
        assert actions == []

    def test_single_position_breach_suggests_reduce(self) -> None:
        """A holding exceeding the single-position limit gets a reduce action."""
        config = PortfolioConfig(
            currency="USD",
            holdings=[
                Holding(ticker="AAPL", shares=100, avg_cost=150.0),
                Holding(ticker="JPM", shares=10, avg_cost=140.0),
            ],
            limits=PortfolioLimits(max_single_position_pct=15.0, max_sector_pct=100.0),
        )
        calc = RiskCalculator(config)
        # AAPL: 100*180=18000, JPM: 10*160=1600, total=19600
        # AAPL weight: 18000/19600 ≈ 91.8% — far above 15%
        snap = calc.build_snapshot({"AAPL": 180.0, "JPM": 160.0})
        exposure = calc.build_exposure(snap)
        actions = calc.suggest_rebalance(snap, exposure)

        aapl_actions = [a for a in actions if a.ticker == "AAPL"]
        assert len(aapl_actions) == 1
        assert aapl_actions[0].action == "reduce"
        assert aapl_actions[0].shares_delta < 0
        assert "single position" in aapl_actions[0].reason.lower() or "exceeds" in aapl_actions[0].reason.lower()

    def test_sector_breach_reduces_largest_first(self) -> None:
        """Sector breach reduces the largest position in that sector first."""
        config = PortfolioConfig(
            currency="USD",
            holdings=[
                Holding(ticker="AAPL", shares=100, avg_cost=150.0),
                Holding(ticker="MSFT", shares=50, avg_cost=300.0),
                Holding(ticker="JPM", shares=10, avg_cost=140.0),
            ],
            limits=PortfolioLimits(max_single_position_pct=100.0, max_sector_pct=40.0),
        )
        calc = RiskCalculator(config)
        # AAPL: 100*180=18000, MSFT: 50*350=17500, JPM: 10*160=1600
        # total=37100. Tech: 35500/37100 ≈ 95.7%, far above 40%
        snap = calc.build_snapshot({"AAPL": 180.0, "MSFT": 350.0, "JPM": 160.0})
        exposure = calc.build_exposure(snap)
        actions = calc.suggest_rebalance(snap, exposure)

        tech_actions = [a for a in actions if a.ticker in ("AAPL", "MSFT")]
        assert len(tech_actions) > 0
        # All tech actions should be "reduce"
        assert all(a.action == "reduce" for a in tech_actions)
        assert all(a.shares_delta < 0 for a in tech_actions)

    def test_sector_breach_respects_minimum_weight(self) -> None:
        """Sector reduction should not push a position below 5% weight."""
        config = PortfolioConfig(
            currency="USD",
            holdings=[
                Holding(ticker="AAPL", shares=50, avg_cost=150.0),
                Holding(ticker="MSFT", shares=5, avg_cost=300.0),  # small position
                Holding(ticker="JPM", shares=200, avg_cost=140.0),
            ],
            limits=PortfolioLimits(max_single_position_pct=100.0, max_sector_pct=20.0),
        )
        calc = RiskCalculator(config)
        # AAPL: 50*180=9000, MSFT: 5*350=1750, JPM: 200*160=32000
        # total=42750. Tech: 10750/42750 ≈ 25.1%, above 20%
        # MSFT weight: 1750/42750 ≈ 4.1% — already below 5% floor
        snap = calc.build_snapshot({"AAPL": 180.0, "MSFT": 350.0, "JPM": 160.0})
        exposure = calc.build_exposure(snap)
        actions = calc.suggest_rebalance(snap, exposure)

        # MSFT should NOT be reduced (below 5% floor)
        msft_actions = [a for a in actions if a.ticker == "MSFT"]
        assert len(msft_actions) == 0

    def test_empty_portfolio_returns_empty(self) -> None:
        config = PortfolioConfig(currency="USD", holdings=[])
        calc = RiskCalculator(config)
        snap = calc.build_snapshot({})
        exposure = calc.build_exposure(snap)
        actions = calc.suggest_rebalance(snap, exposure)
        assert actions == []

    def test_rebalance_action_fields(self) -> None:
        """Verify RebalanceAction model fields."""
        action = RebalanceAction(
            ticker="AAPL",
            action="reduce",
            shares_delta=-10.0,
            reason="Over limit",
        )
        assert action.ticker == "AAPL"
        assert action.action == "reduce"
        assert action.shares_delta == -10.0
        assert action.reason == "Over limit"

    def test_no_duplicate_actions_for_single_and_sector_breach(self) -> None:
        """A holding breaching both single-position and sector limits gets only one action."""
        config = PortfolioConfig(
            currency="USD",
            holdings=[
                Holding(ticker="AAPL", shares=100, avg_cost=150.0),
                Holding(ticker="JPM", shares=10, avg_cost=140.0),
            ],
            limits=PortfolioLimits(max_single_position_pct=15.0, max_sector_pct=20.0),
        )
        calc = RiskCalculator(config)
        snap = calc.build_snapshot({"AAPL": 180.0, "JPM": 160.0})
        exposure = calc.build_exposure(snap)
        actions = calc.suggest_rebalance(snap, exposure)

        aapl_actions = [a for a in actions if a.ticker == "AAPL"]
        # Should have exactly one action (single-position takes priority, sector skips duplicate)
        assert len(aapl_actions) == 1


# ---------------------------------------------------------------------------
# suggest_additions
# ---------------------------------------------------------------------------


class TestSuggestAdditions:
    def test_no_correlation_returns_first_three(self) -> None:
        """Without correlation data, returns first 3 candidates."""
        config = PortfolioConfig(
            currency="USD",
            holdings=[Holding(ticker="AAPL", shares=10, avg_cost=150.0)],
        )
        calc = RiskCalculator(config)
        snap = calc.build_snapshot({"AAPL": 180.0})

        candidates = ["XOM", "JNJ", "KO", "DIS"]
        actions = calc.suggest_additions(candidates, snap)

        assert len(actions) == 3
        assert all(a.action == "add" for a in actions)
        assert [a.ticker for a in actions] == ["XOM", "JNJ", "KO"]

    def test_prefers_low_correlation_candidates(self) -> None:
        """Candidates with lower average correlation rank first."""
        config = PortfolioConfig(
            currency="USD",
            holdings=[
                Holding(ticker="AAPL", shares=50, avg_cost=150.0),
                Holding(ticker="MSFT", shares=50, avg_cost=300.0),
            ],
        )
        calc = RiskCalculator(config)
        snap = calc.build_snapshot({"AAPL": 180.0, "MSFT": 350.0})

        corr = CorrelationResult(
            portfolio_beta=1.0,
            correlation_avg=0.5,
            betas={"AAPL": 1.1, "MSFT": 1.2},
            correlation_matrix={
                "AAPL": {"MSFT": 0.9, "XOM": 0.2, "JNJ": 0.8, "KO": 0.3},
                "MSFT": {"AAPL": 0.9, "XOM": 0.1, "JNJ": 0.7, "KO": 0.4},
                "XOM": {"AAPL": 0.2, "MSFT": 0.1},
                "JNJ": {"AAPL": 0.8, "MSFT": 0.7},
                "KO": {"AAPL": 0.3, "MSFT": 0.4},
            },
        )

        candidates = ["XOM", "JNJ", "KO"]
        actions = calc.suggest_additions(candidates, snap, corr_result=corr)

        assert len(actions) == 3
        # XOM has lowest avg corr: (0.2+0.1)/2 = 0.15
        # KO: (0.3+0.4)/2 = 0.35
        # JNJ: (0.8+0.7)/2 = 0.75
        assert actions[0].ticker == "XOM"
        assert actions[1].ticker == "KO"
        assert actions[2].ticker == "JNJ"

    def test_skips_existing_holdings(self) -> None:
        """Candidates already held should be excluded."""
        config = PortfolioConfig(
            currency="USD",
            holdings=[Holding(ticker="AAPL", shares=10, avg_cost=150.0)],
        )
        calc = RiskCalculator(config)
        snap = calc.build_snapshot({"AAPL": 180.0})

        corr = CorrelationResult(
            portfolio_beta=1.0,
            correlation_avg=0.5,
            betas={"AAPL": 1.1},
            correlation_matrix={
                "AAPL": {"XOM": 0.3},
                "XOM": {"AAPL": 0.3},
            },
        )

        candidates = ["AAPL", "XOM"]
        actions = calc.suggest_additions(candidates, snap, corr_result=corr)

        tickers = [a.ticker for a in actions]
        assert "AAPL" not in tickers
        assert "XOM" in tickers

    def test_returns_at_most_three(self) -> None:
        """Never returns more than 3 suggestions."""
        config = PortfolioConfig(
            currency="USD",
            holdings=[Holding(ticker="AAPL", shares=10, avg_cost=150.0)],
        )
        calc = RiskCalculator(config)
        snap = calc.build_snapshot({"AAPL": 180.0})

        candidates = ["XOM", "JNJ", "KO", "DIS", "WMT"]
        actions = calc.suggest_additions(candidates, snap)
        assert len(actions) <= 3
