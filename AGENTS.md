# AGENTS.md

## Project Overview

Tracer - Investing agent for discovering alpha in global security markets.
Core capabilities: cross-market alpha discovery, contrarian signal detection.

## Tech Stack

- Language: Python 3.12+
- Package manager: uv
- Linter/Formatter: ruff
- Type checker: pyright
- Test: pytest
- LLM: multi-provider (Claude, OpenAI, Gemini, etc.)
- Data: multi-source (Finnhub, yfinance, FRED, FMP, etc.)

## Architecture

### Adapter + Capability Registry Pattern

Both LLM and Data layers use the same pattern: adapters register capabilities,
registry routes requests by capability. Agents never reference a specific source directly.

```
Agent: "I need Price data for AAPL"
  → Registry.get(Price)
  → Returns FinnhubAdapter (primary) or YfinanceAdapter (fallback)
```

### LLM Layer

```
llm/
├── base.py            # LLMProvider protocol (Chat, StructuredOutput, Streaming)
├── registry.py        # role → adapter routing, fallback chain
└── adapters/
    ├── claude.py      # ClaudeAdapter
    ├── openai.py      # OpenAIAdapter
    └── gemini.py      # GeminiAdapter
```

Each adapter registers its capabilities. Prototype defaults to Claude for all roles.
Expand per-role assignment via config when needed.

Roles and default assignments (overridable via config):

| Role       | Description                          | Default          |
|------------|--------------------------------------|------------------|
| researcher | Gather and summarize market data     | Claude Sonnet    |
| analyst    | Deep financial/cross-market analysis | Claude Opus      |
| strategist | Investment decision and signal gen   | Claude Opus      |
| reporter   | Summary and report generation        | Claude Haiku     |

### Data Layer

```
data/
├── base.py            # Capability protocols (Price, Fundamental, Macro, News, Alternative)
├── registry.py        # capability → adapter routing, fallback chain
└── adapters/
    ├── finnhub.py     # Price, News, Alternative (insider, congress)
    ├── yfinance.py    # Price, Fundamental
    ├── fred.py        # Macro
    └── fmp.py         # Fundamental, Alternative (SEC filings)
```

Each adapter is a single class with one client, registering multiple capabilities.
Registry auto-routes by capability with fallback:

```python
class FinnhubAdapter:
    capabilities = [Price, News, Insider, Congress]

    def __init__(self, api_key: str):
        self.client = FinnhubClient(api_key)

    async def get_price(self, ticker: str) -> float: ...
    async def get_news(self, ticker: str) -> list[News]: ...

# Agents request by capability, not by source
registry.get(Price)          # → FinnhubAdapter (primary)
registry.get(Price, "yf")    # → YfinanceAdapter (explicit)
```

Default capability routing (overridable via config):

| Capability    | Primary    | Fallback       |
|---------------|------------|----------------|
| Price/OHLCV   | Finnhub    | yfinance       |
| Fundamental   | Finnhub    | FMP, yfinance  |
| Macro         | FRED       | World Bank     |
| News/Sentiment| Finnhub    | NewsAPI, GDELT |
| Alternative   | Finnhub    | SEC EDGAR      |

API key missing → adapter auto-skipped. Fallback kicks in transparently.

### Storage

DuckDB single-file database (`tracer.db`). Append-only for market data, analytical queries optimized.

```
DuckDB (tracer.db)
├── prices          - OHLCV time series (daily append)
├── fundamentals    - valuation, financial statements (quarterly append)
├── macro           - economic indicators (monthly append)
├── news            - articles + sentiment scores (daily append)
├── alternative     - insider trades, congressional trades, etc. (event append)
├── signals         - generated signal history
├── reports         - analysis report metadata
└── agent_logs      - agent execution logs
```

Also serves as API cache to reduce rate limit pressure.
Export to Parquet for backup/sharing.

### Rate Limits (reference)

| Source     | Free Tier Limit     | Notes                          |
|------------|--------------------|---------------------------------|
| Finnhub    | 60 req/min         | Stable, well-documented         |
| yfinance   | ~2000 req/hr (est) | Unofficial, can get IP-blocked  |
| FRED       | 120 req/min        | Generous                        |
| FMP        | 250 req/day        | Low; use as fallback            |

yfinance: use only for historical data backfill. Avoid repeated real-time calls.

## Project Structure

```
tracer/
├── AGENTS.md
├── pyproject.toml
├── src/
│   └── tracer/
│       ├── __init__.py
│       ├── agents/            # Agent roles (researcher, analyst, strategist, reporter)
│       ├── llm/               # LLM adapter + capability registry
│       │   ├── base.py        # LLMProvider protocol
│       │   ├── registry.py    # Role → adapter routing
│       │   └── adapters/
│       │       ├── claude.py
│       │       ├── openai.py
│       │       └── gemini.py
│       ├── data/              # Data adapter + capability registry
│       │   ├── base.py        # Capability protocols (Price, Fundamental, etc.)
│       │   ├── registry.py    # Capability → adapter routing, fallback
│       │   └── adapters/
│       │       ├── finnhub.py   # Price, News, Alternative
│       │       ├── yfinance.py  # Price, Fundamental
│       │       ├── fred.py      # Macro
│       │       └── fmp.py       # Fundamental, Alternative
│       ├── models/            # Domain models (Stock, Signal, Report, etc.)
│       ├── storage/           # DuckDB persistence layer
│       │   ├── db.py          # Connection management
│       │   └── tables.py      # Schema definitions
│       └── config/            # Configuration loader
├── tests/
│   ├── agents/
│   ├── llm/
│   ├── data/
│   ├── storage/
│   └── strategies/
├── skills/                    # Agent skill definitions (SKILL.md per skill)
└── scripts/                   # CLI entry points, one-off utilities
```

## Agent Pipeline (Tracer Cycle)

```
Screening → Macro Regime → Cross-Market Discovery → Consensus Mapping
    → Contrarian Detection → Conviction Scoring → Alpha Report
```

### Step 1: Universe Screening
- Filter global markets by region, sector, market cap, liquidity
- Narrow down analysis targets from thousands to actionable set
- Agent: **researcher** | Data: PriceProvider, FundamentalProvider

### Step 2: Macro Regime Detection
- Determine current market regime: risk-on / risk-off / transition
- Analyze interest rates, inflation, GDP trends, currency movements
- Regime determines which strategies and sectors to prioritize
- Agent: **analyst** | Data: MacroProvider

### Step 3: Cross-Market Discovery (core alpha)
- Find information asymmetry across global markets
- Detect leading indicators in one market that predict another
- Example: Korea semiconductor exports → US AI stock forward indicator
- Example: China property regulation → commodity demand → AUD weakness → BHP earnings
- Agent: **analyst** | Data: PriceProvider, MacroProvider, NewsProvider, AlternativeProvider

### Step 4: Consensus Mapping
- Collect what the market currently believes
- Analyst ratings, news sentiment, institutional positioning (13F), insider trades
- Build a "consensus view" for each target
- Agent: **researcher** | Data: NewsProvider, AlternativeProvider, FundamentalProvider

### Step 5: Contrarian Detection (core alpha)
- Compare Step 3 findings against Step 4 consensus
- Find where consensus is wrong, late, or ignoring signals
- Identify: oversold with improving fundamentals, overhyped with deteriorating data, ignored catalysts
- Agent: **strategist** | Data: all providers

### Step 6: Conviction Scoring
- Score each signal by strength, time horizon, and risk
- Factors: data quality, signal convergence, historical hit rate, downside scenario
- Output: ranked list of high-conviction ideas with risk assessment
- Agent: **strategist**

### Step 7: Alpha Report
- Generate actionable investment report
- Contents: thesis, supporting evidence, contrarian angle, risk factors, timeline, position sizing suggestion
- Format: "What the market doesn't know yet" narrative
- Agent: **reporter**

## Coding Conventions

- Type hints on all public functions.
- Docstrings only where logic is non-obvious.
- Use `async/await` for I/O-bound operations (API calls).
- All provider implementations must implement the corresponding interface.
- Config-driven: provider selection, model assignment, API keys via environment variables.
- Keep files under 300 LOC; split when exceeded.
- Tests mirror `src/` structure. Minimum: one test per provider, one per agent.

## Git Conventions

- Conventional Commits: `feat|fix|refactor|build|ci|chore|docs|style|perf|test`
- Keep commits atomic and focused.
- Branch naming: `feat/<name>`, `fix/<name>`, `refactor/<name>`
- Do not commit API keys or secrets. Use `.env` (gitignored).

## Development Workflow

1. Read this file before starting any work.
2. Run `uv sync` to install dependencies.
3. Run `ruff check . && ruff format --check .` before committing.
4. Run `pytest` before pushing.
5. Run `pyright` for type checking.
