"""RiskCalculator — portfolio snapshot, exposure, limits, and position sizing."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from qracer.config.models import PortfolioConfig
from qracer.risk.correlation import (
    CorrelationEngine,
    CorrelationResult,
    correlation_adjustment,
)
from qracer.risk.models import (
    ExposureBreakdown,
    HoldingSnapshot,
    PortfolioSnapshot,
    RiskAssessment,
)

logger = logging.getLogger(__name__)

# Hardcoded sector fallback for tickers without a FundamentalProvider.
_TICKER_SECTOR: dict[str, str] = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "GOOG": "Technology",
    "AMZN": "Consumer Discretionary",
    "META": "Technology",
    "NVDA": "Technology",
    "TSLA": "Consumer Discretionary",
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "PFE": "Healthcare",
    "XOM": "Energy",
    "CVX": "Energy",
    "TSM": "Technology",
    "TSMC": "Technology",
    "V": "Financials",
    "MA": "Financials",
    "WMT": "Consumer Staples",
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "DIS": "Communication Services",
    "NFLX": "Communication Services",
    "T": "Communication Services",
}


class SectorResolver:
    """Resolves ticker → sector with dynamic lookup and in-memory cache.

    Tries FundamentalProvider first (if a DataRegistry is available),
    caches results, and falls back to the hardcoded mapping.
    """

    def __init__(self, data_registry: Any | None = None) -> None:
        self._registry = data_registry
        self._cache: dict[str, str] = {}

    def get_sector(self, ticker: str) -> str:
        """Return sector for a ticker, using cache → hardcoded fallback."""
        upper = ticker.upper()
        if upper in self._cache:
            return self._cache[upper]
        sector = _TICKER_SECTOR.get(upper, "Unknown")
        self._cache[upper] = sector
        return sector

    async def get_sector_async(self, ticker: str) -> str:
        """Return sector with async FundamentalProvider lookup + fallback.

        Tries the FundamentalProvider's ``sector`` field first.  On any
        failure falls back to :meth:`get_sector` (hardcoded + cache).
        """
        upper = ticker.upper()
        if upper in self._cache:
            return self._cache[upper]

        if self._registry is not None:
            try:
                from qracer.data.providers import FundamentalProvider

                data = await self._registry.async_get_with_fallback(
                    FundamentalProvider, "get_fundamentals", upper
                )
                if data.sector:
                    self._cache[upper] = data.sector
                    return data.sector
            except Exception:
                logger.debug("Dynamic sector lookup failed for %s, using fallback", upper)

        return self.get_sector(upper)


# Module-level convenience (backwards-compatible).
_default_resolver = SectorResolver()


def get_sector(ticker: str) -> str:
    """Return sector for a ticker, defaulting to 'Unknown'."""
    return _default_resolver.get_sector(ticker)


class RiskCalculator:
    """Builds portfolio snapshots and performs risk calculations."""

    def __init__(
        self,
        config: PortfolioConfig,
        *,
        peak_value: float = 0.0,
        sector_resolver: SectorResolver | None = None,
        correlation_engine: CorrelationEngine | None = None,
    ) -> None:
        self._config = config
        self._peak_value = peak_value
        self._sectors = sector_resolver or _default_resolver
        self._correlation_engine = correlation_engine
        self._last_correlation: CorrelationResult | None = None

    def build_snapshot(self, prices: dict[str, float]) -> PortfolioSnapshot:
        """Build a portfolio snapshot from current prices.

        Args:
            prices: Mapping of ticker -> current price.

        Returns:
            PortfolioSnapshot with computed market values and weights.
        """
        holdings: list[HoldingSnapshot] = []
        total_value = 0.0

        # First pass: compute market values.
        holding_data: list[tuple[str, float, float, float, float]] = []
        for h in self._config.holdings:
            price = prices.get(h.ticker)
            if price is None:
                logger.warning("Price unavailable for %s, skipping in snapshot", h.ticker)
                continue
            market_value = h.shares * price
            total_value += market_value
            holding_data.append((h.ticker, h.shares, h.avg_cost, price, market_value))

        # Second pass: compute weights and PnL.
        for ticker, shares, avg_cost, current_price, market_value in holding_data:
            weight_pct = (market_value / total_value * 100.0) if total_value > 0 else 0.0
            cost_basis = shares * avg_cost
            unrealized_pnl = market_value - cost_basis
            unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100.0) if cost_basis > 0 else 0.0

            holdings.append(
                HoldingSnapshot(
                    ticker=ticker,
                    shares=shares,
                    avg_cost=avg_cost,
                    current_price=current_price,
                    market_value=market_value,
                    weight_pct=round(weight_pct, 2),
                    unrealized_pnl=round(unrealized_pnl, 2),
                    unrealized_pnl_pct=round(unrealized_pnl_pct, 2),
                )
            )

        return PortfolioSnapshot(
            holdings=holdings,
            total_value=round(total_value, 2),
            currency=self._config.currency,
            as_of=datetime.now(),
        )

    @staticmethod
    def _exposure_from_sector_values(
        sector_values: dict[str, float],
        total: float,
        *,
        portfolio_beta: float | None = None,
        correlation_avg: float | None = None,
        correlation_data_unavailable: bool = False,
    ) -> ExposureBreakdown:
        """Build an ExposureBreakdown from raw sector → market-value mapping."""
        sector_weights: dict[str, float] = {}
        for sector, value in sector_values.items():
            sector_weights[sector] = round(value / total * 100.0, 2) if total > 0 else 0.0

        if sector_weights:
            top_sector = max(sector_weights, key=lambda s: sector_weights[s])
            top_sector_pct = sector_weights[top_sector]
        else:
            top_sector = "N/A"
            top_sector_pct = 0.0

        return ExposureBreakdown(
            sector_weights=sector_weights,
            top_sector=top_sector,
            top_sector_pct=top_sector_pct,
            portfolio_beta=portfolio_beta,
            correlation_avg=correlation_avg,
            correlation_data_unavailable=correlation_data_unavailable,
        )

    def build_exposure(self, snapshot: PortfolioSnapshot) -> ExposureBreakdown:
        """Build exposure breakdown from a portfolio snapshot."""
        sector_values: dict[str, float] = {}
        for h in snapshot.holdings:
            sector = self._sectors.get_sector(h.ticker)
            sector_values[sector] = sector_values.get(sector, 0.0) + h.market_value

        return self._exposure_from_sector_values(sector_values, snapshot.total_value)

    async def build_exposure_async(self, snapshot: PortfolioSnapshot) -> ExposureBreakdown:
        """Build exposure breakdown with async sector lookup and correlation/beta."""
        sector_values: dict[str, float] = {}
        for h in snapshot.holdings:
            sector = await self._sectors.get_sector_async(h.ticker)
            sector_values[sector] = sector_values.get(sector, 0.0) + h.market_value

        # Correlation/beta computation.
        portfolio_beta: float | None = None
        correlation_avg: float | None = None
        correlation_data_unavailable = False

        if self._correlation_engine is not None and snapshot.holdings:
            corr_result = await self._correlation_engine.compute(snapshot.holdings)
            if corr_result is not None:
                self._last_correlation = corr_result
                portfolio_beta = corr_result.portfolio_beta
                correlation_avg = corr_result.correlation_avg
            else:
                correlation_data_unavailable = True

        return self._exposure_from_sector_values(
            sector_values,
            snapshot.total_value,
            portfolio_beta=portfolio_beta,
            correlation_avg=correlation_avg,
            correlation_data_unavailable=correlation_data_unavailable,
        )

    def check_limits(self, snapshot: PortfolioSnapshot, exposure: ExposureBreakdown) -> list[str]:
        """Check portfolio limits and return descriptions of breached limits."""
        breached: list[str] = []
        limits = self._config.limits

        for h in snapshot.holdings:
            if h.weight_pct > limits.max_single_position_pct:
                breached.append(
                    f"{h.ticker} weight {h.weight_pct:.1f}% exceeds "
                    f"max single position limit {limits.max_single_position_pct:.1f}%"
                )

        for sector, pct in exposure.sector_weights.items():
            if pct > limits.max_sector_pct:
                breached.append(
                    f"Sector '{sector}' weight {pct:.1f}% exceeds "
                    f"max sector limit {limits.max_sector_pct:.1f}%"
                )

        return breached

    def size_position(
        self,
        ticker: str,
        conviction: int,
        snapshot: PortfolioSnapshot,
        corr_result: CorrelationResult | None = None,
    ) -> float:
        """Compute position size as % of portfolio based on conviction.

        When *corr_result* is provided (or previously cached via
        ``build_exposure_async``), the allocation is reduced if the
        ticker is highly correlated with existing large positions.
        """
        if conviction >= 8:
            base_pct = 3.0 + (conviction - 8) * 1.0
        elif conviction >= 5:
            base_pct = 1.0 + (conviction - 5) * 1.0
        else:
            base_pct = 0.5 + (conviction - 1) * (0.5 / 3)

        sector = self._sectors.get_sector(ticker)
        exposure = self._compute_sector_weight(sector, snapshot)
        sector_limit = self._config.limits.max_sector_pct
        sector_headroom = max(0.0, sector_limit - exposure)

        if sector_headroom < base_pct:
            base_pct = sector_headroom

        # Apply correlation-based adjustment.
        effective_corr = corr_result or self._last_correlation
        if effective_corr is not None:
            adj = correlation_adjustment(ticker, snapshot.holdings, effective_corr)
            base_pct *= adj

        max_pos = self._config.limits.max_single_position_pct
        if base_pct > max_pos:
            base_pct = max_pos

        return round(max(base_pct, 0.0), 2)

    def _compute_sector_weight(self, sector: str, snapshot: PortfolioSnapshot) -> float:
        """Compute the current total weight of a sector in the portfolio."""
        total = 0.0
        for h in snapshot.holdings:
            if self._sectors.get_sector(h.ticker) == sector:
                total += h.weight_pct
        return total

    def update_peak(self, current_value: float) -> float:
        """Update and return the peak portfolio value."""
        if current_value > self._peak_value:
            self._peak_value = current_value
        return self._peak_value

    @property
    def peak_value(self) -> float:
        """Current recorded peak value."""
        return self._peak_value

    def compute_drawdown(self, current_value: float) -> float:
        """Compute current drawdown as a percentage."""
        if self._peak_value <= 0 or current_value >= self._peak_value:
            return 0.0
        return (self._peak_value - current_value) / self._peak_value * 100.0

    def _build_assessment(
        self,
        snapshot: PortfolioSnapshot,
        exposure: ExposureBreakdown,
    ) -> RiskAssessment:
        """Shared logic for assess/assess_async: limits, drawdown, warnings."""
        breached = self.check_limits(snapshot, exposure)

        self.update_peak(snapshot.total_value)
        drawdown_pct = self.compute_drawdown(snapshot.total_value)
        drawdown_alert = drawdown_pct > self._config.limits.max_drawdown_alert_pct

        warnings: list[str] = []
        if exposure.correlation_data_unavailable:
            warnings.append(
                "Correlation/beta data unavailable — risk assessment "
                "proceeded without correlation adjustments"
            )

        return RiskAssessment(
            snapshot=snapshot,
            exposure=exposure,
            limits_breached=breached,
            warnings=warnings,
            max_drawdown_alert=drawdown_alert,
            current_drawdown_pct=round(drawdown_pct, 2),
            peak_value=round(self._peak_value, 2),
        )

    def assess(self, prices: dict[str, float]) -> RiskAssessment:
        """Run a full risk assessment."""
        snapshot = self.build_snapshot(prices)
        exposure = self.build_exposure(snapshot)
        return self._build_assessment(snapshot, exposure)

    async def assess_async(self, prices: dict[str, float]) -> RiskAssessment:
        """Run a full risk assessment with async correlation/beta computation."""
        snapshot = self.build_snapshot(prices)
        exposure = await self.build_exposure_async(snapshot)
        return self._build_assessment(snapshot, exposure)
