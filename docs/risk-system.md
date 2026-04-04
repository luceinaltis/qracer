# Risk System

Portfolio-aware risk management that integrates with the ResearchPipeline to produce sized recommendations.

## Portfolio Model

Holdings are loaded from `.qracer/portfolio.toml`:

```toml
[portfolio]
currency = "USD"

[[portfolio.holdings]]
ticker = "AAPL"
shares = 100
avg_cost = 165.00

[[portfolio.holdings]]
ticker = "TSMC"
shares = 200
avg_cost = 140.00

[portfolio.limits]
max_single_position_pct = 15    # max % of portfolio in one name
max_sector_pct = 40             # max % in one sector
max_drawdown_alert_pct = 10     # alert when portfolio drawdown exceeds this
```

Real-time P&L tracking uses LivePipeline price feeds. Sector and geography classification derived from fundamental data.

## Exposure Breakdown

The risk module maintains a live view of portfolio exposure:

| Dimension | Calculation | Source |
|-----------|------------|--------|
| Sector concentration | Market value per GICS sector / total | FundamentalProvider |
| Geography exposure | Revenue-weighted country allocation | FundamentalProvider |
| Beta | Portfolio-weighted beta vs benchmark | PriceProvider (90-day) |
| Correlation matrix | Pairwise correlation between holdings | PriceProvider (90-day) |

## Risk Metrics

| Metric | Description |
|--------|-------------|
| Portfolio beta | Weighted average beta vs S&P 500 |
| Sharpe ratio | Risk-adjusted return (rolling 90-day) |
| Max drawdown | Largest peak-to-trough decline |
| Current drawdown | Current level vs all-time high |
| Sector concentration | Largest sector weight |
| Correlation risk | Average pairwise correlation (high = clustered risk) |

## Position Sizing

Conviction score (1-10) from the ResearchPipeline maps to a base allocation:

| Conviction | Base Allocation | Description |
|-----------|----------------|-------------|
| 8-10 | 3-5% of portfolio | High conviction |
| 5-7 | 1-3% of portfolio | Moderate conviction |
| 1-4 | 0.5-1% of portfolio | Low conviction / tracking position |

Base allocation is then adjusted by:

1. **Sector exposure** — reduce if sector already near limit
2. **Correlation** — reduce if highly correlated with existing large positions
3. **Volatility** — reduce for high-vol names to normalize risk contribution
4. **Hard limits** — never exceed `max_single_position_pct` from `portfolio.toml`

## Integration with Pipeline

The risk module is consulted at ResearchPipeline Step 8 (Risk Check):

```text
Step 7 output (Trade Thesis)
    → Load current portfolio state
    → Calculate exposure impact of proposed position
    → Apply sizing algorithm
    → Enforce hard limits
    → Output: sized recommendation or rejection with reason
```

If a recommendation would breach portfolio limits, the risk module either reduces the size or flags it with a warning rather than silently blocking.

## Autonomous Mode Integration

During autonomous monitoring (autonomous mode, planned), the risk module triggers alerts when:

- Portfolio drawdown exceeds `max_drawdown_alert_pct`
- Single position grows beyond `max_single_position_pct` (due to price movement)
- Sector concentration drifts above `max_sector_pct`
- Correlation clustering increases significantly (new positions or market regime change)
