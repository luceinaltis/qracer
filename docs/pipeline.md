# Pipeline

Tracer has two pipelines. The IntentRouter selects which one handles each query.

## LivePipeline (QuickPath)

For market-hours conversational use. Target: **< 5 seconds end-to-end**.

```text
Query → IntentRouter → Tool (1-2 max) → Format → Response
```

### Flow

1. **IntentRouter** classifies the query into an intent type (single LLM call, or rule-based for common patterns like "price of X")
2. **Tool dispatch**: 1-2 data tools execute in parallel. Only hot/warm-tier adapters.
3. **Format**: Template-based response — no LLM call for simple lookups, one LLM call for summaries.

### Intent → Tool Mapping

| Intent | Example | Tools | LLM Calls |
|---|---|---|---|
| `price_check` | "What's AAPL at?" | price_quote | 0 |
| `quick_news` | "Any news on TSLA?" | news (cached) | 0–1 |
| `comparison` | "AAPL vs MSFT P/E" | fundamentals | 0 |
| `follow_up` | "What about Samsung?" | resolved from context → appropriate tool | 0–1 |
| `opinion` | "Is NVDA overvalued?" | price_quote, fundamentals | 1 |
| `macro_check` | "Where's the 10Y at?" | macro | 0 |
| `alert_set` | "Tell me if AAPL drops below 170" | alert_register | 0 |

### Latency Budget

| Step | Budget |
|---|---|
| Intent classification | < 500ms (rule-based) or < 1.5s (LLM) |
| Data fetch (parallel) | < 2s |
| Response formatting | < 500ms (template) or < 1.5s (LLM) |
| **Total** | **< 5s** |

If data fetch exceeds 2s, return partial data with a staleness note.

## ResearchPipeline (DeepPath)

Full 7-step analysis. Runs async — user is notified on completion.

```text
Screening → Macro Regime → Cross-Market Discovery → Consensus Mapping
    → Contrarian Detection → Conviction Scoring → Alpha Report
```

### Step 1: Universe Screening
- Filter by region, sector, market cap, liquidity
- Agent: **researcher** | Data: PriceProvider, FundamentalProvider

### Step 2: Macro Regime Detection
- Determine regime: risk-on / risk-off / transition
- Analyze rates, inflation, GDP, currencies
- Agent: **analyst** | Data: MacroProvider

### Step 3: Cross-Market Discovery (core alpha)
- Find information asymmetry across global markets
- Detect leading indicators: Korea semi exports → US AI stocks, China property → commodities → AUD → BHP
- Agent: **analyst** | Data: PriceProvider, MacroProvider, NewsProvider, AlternativeProvider

### Step 4: Consensus Mapping
- Collect analyst ratings, news sentiment, institutional positioning, insider trades
- Build consensus view per target
- Agent: **researcher** | Data: NewsProvider, AlternativeProvider, FundamentalProvider

### Step 5: Contrarian Detection (core alpha)
- Compare Step 3 against Step 4 consensus
- Find where consensus is wrong, late, or ignoring signals
- Agent: **strategist** | Data: all providers

### Step 6: Conviction Scoring
- Score by strength, time horizon, risk
- Output: ranked list with risk assessment
- Agent: **strategist**

### Step 7: Alpha Report
- Generate full report: thesis, evidence, contrarian angle, risk factors, timeline
- Agent: **reporter**

### Trigger Conditions

ResearchPipeline runs when **any** of these are true:

| Trigger | Example |
|---|---|
| Explicit request | "Run a full analysis on TSMC" |
| `/deep` command | `/deep AAPL` |
| `deep_dive` intent | "Give me everything on Samsung" |
| `alpha_hunt` intent | "Where's the hidden alpha right now?" |
| Scheduled | Nightly batch run (if configured) |

Everything else goes through LivePipeline.

## Error Handling

| Failure | Behavior |
|---|---|
| Tool timeout in LivePipeline | Return partial data with caveat |
| Single tool fails | Exclude from evidence; note as "data unavailable" |
| ≥2 tools fail in ResearchPipeline | Exit loop early; caveat lists missing data |
| LLM call fails | Retry once; return partial result if still failing |
| Rate limit hit | Serve from cache if available; otherwise caveat |
