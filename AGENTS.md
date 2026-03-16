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

### LLM Provider Abstraction

Each agent role binds to an LLM provider interface, not a concrete model.
Swap providers per role without touching agent logic.

```
LLMProvider (interface)
├── ClaudeProvider
├── OpenAIProvider
├── GeminiProvider
└── ... (extensible)
```

Roles and default assignments (overridable via config):

| Role       | Description                        | Default Provider |
|------------|------------------------------------|------------------|
| researcher | Gather and summarize market data   | Claude Sonnet    |
| analyst    | Deep financial/cross-market analysis | Claude Opus    |
| strategist | Investment decision and signal generation | Claude Opus |
| reporter   | Summary and report generation      | Claude Haiku     |

### Data Provider Abstraction

Each data type binds to a provider interface. Add/swap sources without changing agent logic.

```
DataProvider (interface)
├── PriceProvider      - stock price, OHLCV, historical
├── FundamentalProvider - financial statements, valuation (PE, PEG, PBR, EV/EBITDA)
├── MacroProvider      - interest rates, GDP, CPI, exchange rates
├── NewsProvider       - news articles, sentiment scores
├── AlternativeProvider - insider trading, congressional trades, ESG, SEC filings
└── ... (extensible)
```

Default source mapping (overridable via config):

| Data Type     | Primary Source | Fallback       |
|---------------|---------------|----------------|
| Price/OHLCV   | Finnhub       | yfinance       |
| Fundamentals  | Finnhub       | FMP, yfinance  |
| Macro         | FRED          | World Bank     |
| News/Sentiment| Finnhub       | NewsAPI, GDELT |
| Alternative   | Finnhub       | SEC EDGAR      |

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
│       ├── llm/               # LLM provider abstraction
│       │   ├── base.py        # LLMProvider interface
│       │   ├── claude.py
│       │   ├── openai.py
│       │   └── gemini.py
│       ├── data/              # Data provider abstraction
│       │   ├── base.py        # DataProvider interfaces
│       │   ├── finnhub.py
│       │   ├── yfinance.py
│       │   ├── fred.py
│       │   └── fmp.py
│       ├── models/            # Domain models (Stock, Signal, Report, etc.)
│       ├── strategies/        # Alpha discovery & contrarian signal strategies
│       └── config/            # Configuration and provider registry
├── tests/
│   ├── agents/
│   ├── llm/
│   ├── data/
│   └── strategies/
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
