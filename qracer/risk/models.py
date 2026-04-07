"""Risk system domain models (Pydantic v2)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class HoldingSnapshot(BaseModel):
    """Point-in-time snapshot of a single holding."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    shares: float
    avg_cost: float
    current_price: float
    market_value: float
    weight_pct: float  # % of total portfolio
    unrealized_pnl: float
    unrealized_pnl_pct: float


class PortfolioSnapshot(BaseModel):
    """Point-in-time snapshot of the entire portfolio."""

    model_config = ConfigDict(frozen=True)

    holdings: list[HoldingSnapshot]
    total_value: float
    currency: str
    as_of: datetime


class ExposureBreakdown(BaseModel):
    """Portfolio exposure analysis."""

    model_config = ConfigDict(frozen=True)

    sector_weights: dict[str, float]  # sector -> % of portfolio
    top_sector: str
    top_sector_pct: float
    portfolio_beta: float | None = None
    correlation_avg: float | None = None
    correlation_data_unavailable: bool = False


class RiskAssessment(BaseModel):
    """Full risk assessment combining snapshot, exposure, and limit checks."""

    snapshot: PortfolioSnapshot
    exposure: ExposureBreakdown
    limits_breached: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    max_drawdown_alert: bool = False
    current_drawdown_pct: float = 0.0
    peak_value: float = 0.0
