# Architecture

## Dual-Mode Design

Tracer operates in two modes based on context:

| | Live Mode | Research Mode |
|---|---|---|
| **When** | Market hours, quick queries | On-demand, explicit request |
| **Latency** | < 5 seconds | 30–120 seconds |
| **LLM calls** | 0–1 | 3–7 |
| **Data** | Real-time quotes, cached fundamentals | Full fetch across all providers |
| **Output** | Short answer, table, or alert | Full analysis report with evidence chain |

Mode is selected automatically by the IntentRouter based on query complexity, but can be forced by the user (`/deep`, `/quick`).

## QuickPath vs DeepPath

```text
QuickPath (Live Mode)
  User query → IntentRouter → 1-2 data tools → format response
  Target: < 5s end-to-end

DeepPath (Research Mode)
  User query → IntentRouter → ResearchPipeline (7-step) → full report
  Target: async, notify on completion
```

QuickPath handles ~80% of market-hours queries: price checks, news summaries, quick comparisons, follow-ups. DeepPath is triggered explicitly or when the query requires cross-market analysis.

## Adapter + Capability Registry Pattern

Adapters register capabilities, registry routes requests by capability. Agents never reference a specific source directly.

```python
class FinnhubAdapter:
    capabilities = [Price, News, Insider, Congress]

    async def get_price(self, ticker: str) -> float: ...
    async def get_news(self, ticker: str) -> list[News]: ...

# Agents request by capability, not by source
registry.get(Price)          # → FinnhubAdapter (primary)
registry.get(Price, "yf")    # → YfinanceAdapter (explicit)
```

### Latency Constraints

Adapters used in QuickPath must meet latency requirements:

| Tier | Max Latency | Used In | Examples |
|------|-------------|---------|----------|
| **hot** | < 2s | QuickPath | Price, cached News |
| **warm** | < 10s | Either | Fundamentals, Macro |
| **cold** | unbounded | DeepPath only | Full backfill, alternative data |

Registry tags each adapter with its tier. QuickPath only dispatches to hot/warm adapters. If a hot adapter times out, the response includes a staleness caveat rather than blocking.

## Real-Time Data

For Live Mode, Tracer needs sub-second price data and streaming news:

| Capability | Preferred Provider | Protocol | Fallback |
|---|---|---|---|
| Real-time quotes | Finnhub | WebSocket | REST polling (5s interval) |
| Streaming news | Finnhub | WebSocket | REST polling (30s interval) |
| Price/OHLCV | Finnhub | REST | yfinance |
| Fundamental | Finnhub | REST | FMP, yfinance |
| Macro | FRED | REST | World Bank |
| News/Sentiment | Finnhub | REST | NewsAPI, GDELT |
| Alternative | Finnhub | REST | SEC EDGAR |

WebSocket connections are opened on session start during market hours and closed on session end. REST fallback activates automatically if WebSocket disconnects.

API key missing → adapter auto-skipped. Fallback kicks in transparently.

## Storage

DuckDB single-file database (`tracer.db`). Append-only for market data, analytical queries optimized.

```text
DuckDB (tracer.db)
├── prices             - OHLCV time series (daily append)
├── prices_intraday    - tick/1min data during live sessions (ephemeral, pruned daily)
├── fundamentals       - valuation, financial statements (quarterly append)
├── macro              - economic indicators (monthly append)
├── news               - articles + sentiment scores (daily append)
├── alternative        - insider trades, congressional trades, etc. (event append)
├── signals            - generated signal history
├── reports            - analysis report metadata
├── agent_logs         - agent execution logs
├── session_index      - session summary metadata + FTS index
├── session_embeddings - session summary embeddings + HNSW index
└── alerts             - active alert rules and trigger history
```

Also serves as API cache to reduce rate limit pressure. Export to Parquet for backup/sharing.

## Rate Limits

| Source | Free Tier Limit | Notes |
|---|---|---|
| Finnhub | 60 req/min | WebSocket preferred for real-time |
| yfinance | ~2000 req/hr (est) | Historical backfill only, avoid real-time |
| FRED | 120 req/min | Generous |
| FMP | 250 req/day | Low; fallback only |
