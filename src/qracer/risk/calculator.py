"""RiskCalculator — portfolio snapshot, exposure, limits, and position sizing."""

from __future__ import annotations

import logging
from datetime import datetime

from qracer.config.models import PortfolioConfig
from qracer.risk.models import (
    ExposureBreakdown,
    HoldingSnapshot,
    PortfolioSnapshot,
    RiskAssessment,
)

logger = logging.getLogger(__name__)

# Hardcoded sector mapping for initial implementation.
# Will be replaced by FundamentalProvider lookups later.
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


def get_sector(ticker: str) -> str:
    """Return sector for a ticker, defaulting to 'Unknown'."""
    return _TICKER_SECTOR.get(ticker.upper(), "Unknown")


class RiskCalculator:
    """Builds portfolio snapshots and performs risk calculations."""

    def __init__(self, config: PortfolioConfig) -> None:
        self._config = config

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

    def build_exposure(self, snapshot: PortfolioSnapshot) -> ExposureBreakdown:
        """Build exposure breakdown from a portfolio snapshot."""
        sector_values: dict[str, float] = {}

        for h in snapshot.holdings:
            sector = get_sector(h.ticker)
            sector_values[sector] = sector_values.get(sector, 0.0) + h.market_value

        total = snapshot.total_value
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
            portfolio_beta=None,
            correlation_avg=None,
        )

    def check_limits(self, snapshot: PortfolioSnapshot, exposure: ExposureBreakdown) -> list[str]:
        """Check portfolio limits and return descriptions of breached limits."""
        breached: list[str] = []
        limits = self._config.limits

        # Check single position limits.
        for h in snapshot.holdings:
            if h.weight_pct > limits.max_single_position_pct:
                breached.append(
                    f"{h.ticker} weight {h.weight_pct:.1f}% exceeds "
                    f"max single position limit {limits.max_single_position_pct:.1f}%"
                )

        # Check sector limits.
        for sector, pct in exposure.sector_weights.items():
            if pct > limits.max_sector_pct:
                breached.append(
                    f"Sector '{sector}' weight {pct:.1f}% exceeds "
                    f"max sector limit {limits.max_sector_pct:.1f}%"
                )

        return breached

    def size_position(self, ticker: str, conviction: int, snapshot: PortfolioSnapshot) -> float:
        """Compute position size as % of portfolio based on conviction.

        Conviction mapping:
            8-10: 3-5% (high conviction)
            5-7:  1-3% (moderate conviction)
            1-4:  0.5-1% (low conviction / tracking)

        The allocation is adjusted down if the ticker's sector is near the
        sector limit. The result never exceeds max_single_position_pct.

        Returns:
            Allocation as a percentage of portfolio (e.g. 3.0 means 3%).
        """
        # Base allocation from conviction.
        if conviction >= 8:
            # Linear interpolation: 8->3%, 9->4%, 10->5%
            base_pct = 3.0 + (conviction - 8) * 1.0
        elif conviction >= 5:
            # Linear interpolation: 5->1%, 6->2%, 7->3%
            base_pct = 1.0 + (conviction - 5) * 1.0
        else:
            # Linear interpolation: 1->0.5%, 2->0.67%, 3->0.83%, 4->1.0%
            base_pct = 0.5 + (conviction - 1) * (0.5 / 3)

        # Adjust for sector exposure proximity to limit.
        sector = get_sector(ticker)
        exposure = self._compute_sector_weight(sector, snapshot)
        sector_limit = self._config.limits.max_sector_pct
        sector_headroom = max(0.0, sector_limit - exposure)

        if sector_headroom < base_pct:
            base_pct = sector_headroom

        # Enforce single position cap.
        max_pos = self._config.limits.max_single_position_pct
        if base_pct > max_pos:
            base_pct = max_pos

        return round(max(base_pct, 0.0), 2)

    def _compute_sector_weight(self, sector: str, snapshot: PortfolioSnapshot) -> float:
        """Compute the current total weight of a sector in the portfolio."""
        total = 0.0
        for h in snapshot.holdings:
            if get_sector(h.ticker) == sector:
                total += h.weight_pct
        return total

    def assess(self, prices: dict[str, float]) -> RiskAssessment:
        """Run a full risk assessment.

        Convenience method that builds snapshot, exposure, and checks limits.
        """
        snapshot = self.build_snapshot(prices)
        exposure = self.build_exposure(snapshot)
        breached = self.check_limits(snapshot, exposure)

        return RiskAssessment(
            snapshot=snapshot,
            exposure=exposure,
            limits_breached=breached,
            max_drawdown_alert=False,  # TODO: implement drawdown tracking
        )
