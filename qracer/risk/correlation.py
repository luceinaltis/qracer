"""Correlation and beta computation for portfolio risk analysis.

Provides a CorrelationEngine that computes:
- Individual stock beta vs a benchmark (S&P 500 proxy)
- Pairwise correlation matrix between holdings
- Portfolio-weighted average beta and average correlation

Uses 90-day rolling windows of daily returns.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from qracer.data.providers import OHLCV
from qracer.risk.models import HoldingSnapshot

logger = logging.getLogger(__name__)

ROLLING_WINDOW_DAYS = 90
BENCHMARK_TICKER = "SPY"
HIGH_CORRELATION_THRESHOLD = 0.7


@dataclass(frozen=True)
class CorrelationResult:
    """Result of correlation/beta analysis for a portfolio."""

    portfolio_beta: float
    correlation_avg: float
    betas: dict[str, float]  # ticker -> beta
    correlation_matrix: dict[str, dict[str, float]]  # ticker -> ticker -> corr


def _daily_returns(bars: list[OHLCV]) -> list[float]:
    """Compute daily log returns from OHLCV bars."""
    if len(bars) < 2:
        return []
    returns = []
    for i in range(1, len(bars)):
        prev_close = bars[i - 1].close
        curr_close = bars[i].close
        if prev_close > 0:
            returns.append(curr_close / prev_close - 1.0)
    return returns


def _align_returns(
    returns_a: list[float], returns_b: list[float]
) -> tuple[list[float], list[float]]:
    """Align two return series to the same length (truncate to shorter)."""
    min_len = min(len(returns_a), len(returns_b))
    return returns_a[:min_len], returns_b[:min_len]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _covariance(xs: list[float], ys: list[float]) -> float:
    """Sample covariance between two aligned series."""
    if len(xs) < 2:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) - 1)


def _variance(xs: list[float]) -> float:
    """Sample variance."""
    if len(xs) < 2:
        return 0.0
    mx = _mean(xs)
    return sum((x - mx) ** 2 for x in xs) / (len(xs) - 1)


def _correlation(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient between two aligned series."""
    cov = _covariance(xs, ys)
    vx, vy = _variance(xs), _variance(ys)
    denom = math.sqrt(vx * vy)
    if denom == 0:
        return 0.0
    return cov / denom


def compute_beta(stock_returns: list[float], benchmark_returns: list[float]) -> float:
    """Compute beta of stock vs benchmark."""
    stock_r, bench_r = _align_returns(stock_returns, benchmark_returns)
    if len(stock_r) < 2:
        return 1.0  # default beta when insufficient data
    var_bench = _variance(bench_r)
    if var_bench == 0:
        return 1.0
    return _covariance(stock_r, bench_r) / var_bench


def compute_correlation_result(
    holdings: list[HoldingSnapshot],
    ohlcv_data: dict[str, list[OHLCV]],
    benchmark_bars: list[OHLCV],
) -> CorrelationResult:
    """Compute portfolio beta and pairwise correlations from OHLCV data.

    Args:
        holdings: Current portfolio holdings with weights.
        ohlcv_data: Mapping of ticker -> OHLCV bars (90-day).
        benchmark_bars: OHLCV bars for the benchmark (SPY).

    Returns:
        CorrelationResult with beta, correlations, and matrix.
    """
    benchmark_returns = _daily_returns(benchmark_bars)

    # Compute per-ticker returns and betas.
    ticker_returns: dict[str, list[float]] = {}
    betas: dict[str, float] = {}

    for h in holdings:
        bars = ohlcv_data.get(h.ticker, [])
        returns = _daily_returns(bars)
        ticker_returns[h.ticker] = returns
        betas[h.ticker] = compute_beta(returns, benchmark_returns)

    # Portfolio-weighted beta.
    total_weight = sum(h.weight_pct for h in holdings)
    if total_weight > 0:
        portfolio_beta = sum(
            betas.get(h.ticker, 1.0) * h.weight_pct / total_weight for h in holdings
        )
    else:
        portfolio_beta = 1.0

    # Pairwise correlation matrix.
    tickers = [h.ticker for h in holdings]
    corr_matrix: dict[str, dict[str, float]] = {}
    pair_correlations: list[float] = []

    for i, t1 in enumerate(tickers):
        corr_matrix[t1] = {}
        for j, t2 in enumerate(tickers):
            if i == j:
                corr_matrix[t1][t2] = 1.0
            elif j < i:
                # Already computed the symmetric pair.
                corr_matrix[t1][t2] = corr_matrix[t2][t1]
            else:
                r1, r2 = _align_returns(ticker_returns.get(t1, []), ticker_returns.get(t2, []))
                corr_val = _correlation(r1, r2)
                corr_matrix[t1][t2] = round(corr_val, 4)
                pair_correlations.append(corr_val)

    correlation_avg = _mean(pair_correlations) if pair_correlations else 0.0

    return CorrelationResult(
        portfolio_beta=round(portfolio_beta, 4),
        correlation_avg=round(correlation_avg, 4),
        betas={t: round(b, 4) for t, b in betas.items()},
        correlation_matrix=corr_matrix,
    )


class CorrelationEngine:
    """Fetches OHLCV data and computes correlation/beta metrics.

    Results are cached daily (keyed by the set of tickers and the date).
    """

    def __init__(self, price_provider: Any = None) -> None:
        self._provider = price_provider
        self._cache: dict[str, CorrelationResult] = {}

    def _cache_key(self, tickers: list[str], as_of: date) -> str:
        sorted_tickers = ",".join(sorted(tickers))
        return f"{sorted_tickers}:{as_of.isoformat()}"

    async def compute(
        self,
        holdings: list[HoldingSnapshot],
        *,
        as_of: date | None = None,
    ) -> CorrelationResult | None:
        """Compute correlation/beta for the given holdings.

        Returns None if no price provider is available or data fetch fails.
        """
        if self._provider is None:
            return None

        if not holdings:
            return None

        today = as_of or date.today()
        tickers = [h.ticker for h in holdings]
        key = self._cache_key(tickers, today)

        if key in self._cache:
            return self._cache[key]

        start = today - timedelta(days=ROLLING_WINDOW_DAYS)
        end = today

        try:
            ohlcv_data: dict[str, list[OHLCV]] = {}
            for ticker in tickers:
                bars = await self._provider.get_ohlcv(ticker, start, end)
                ohlcv_data[ticker] = bars

            benchmark_bars = await self._provider.get_ohlcv(BENCHMARK_TICKER, start, end)
        except Exception:
            logger.warning(
                "Correlation/beta unavailable: failed to fetch OHLCV data for %s. "
                "Risk assessment will proceed without correlation adjustments.",
                [h.ticker for h in holdings],
                exc_info=True,
            )
            return None

        result = compute_correlation_result(holdings, ohlcv_data, benchmark_bars)
        self._cache[key] = result
        return result

    def get_cached(self, tickers: list[str], as_of: date) -> CorrelationResult | None:
        """Return cached result if available, else None."""
        key = self._cache_key(tickers, as_of)
        return self._cache.get(key)

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()


def correlation_adjustment(
    ticker: str,
    holdings: list[HoldingSnapshot],
    corr_result: CorrelationResult | None,
) -> float:
    """Return a multiplier (0.0–1.0) to reduce allocation for correlated positions.

    If the new ticker is highly correlated (>0.7) with large existing
    holdings, the allocation is reduced proportionally.
    """
    if corr_result is None:
        return 1.0

    if ticker not in corr_result.correlation_matrix:
        return 1.0

    ticker_corrs = corr_result.correlation_matrix[ticker]

    # Weight-adjusted high-correlation penalty.
    total_weight = sum(h.weight_pct for h in holdings)
    if total_weight == 0:
        return 1.0

    weighted_high_corr = 0.0
    for h in holdings:
        if h.ticker == ticker:
            continue
        corr_val = abs(ticker_corrs.get(h.ticker, 0.0))
        if corr_val > HIGH_CORRELATION_THRESHOLD:
            weighted_high_corr += (corr_val - HIGH_CORRELATION_THRESHOLD) * (
                h.weight_pct / total_weight
            )

    # Scale: reduce by up to 50% for very high correlation with large positions.
    reduction = min(weighted_high_corr * (1.0 / (1.0 - HIGH_CORRELATION_THRESHOLD)), 1.0)
    return round(max(1.0 - reduction * 0.5, 0.5), 4)
