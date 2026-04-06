# qracer

Conversational investment agent for discovering alpha in global security markets.

Natural language queries → cross-market analysis → actionable alpha reports with sized recommendations.

## Features

- **Dual-mode pipeline** — QuickPath (< 5s) for market-hours queries, DeepPath (9-step) for full research
- **Multi-provider data** — yfinance (built-in), Finnhub, FRED, FMP via plugin system
- **LLM agent pipeline** — Researcher → Analyst → Strategist → Reporter, each with a dedicated role
- **Portfolio-aware risk** — position sizing by conviction, sector limits, drawdown alerts
- **Session memory** — 3-tier architecture (JSONL audit → compressed summaries → DuckDB search index)
- **Multi-ticker comparison** — side-by-side analysis with comparative verdict

## Quick Start

```bash
# Install dependencies
uv sync

# First-time setup (creates ~/.qracer/ config)
qracer install

# Start interactive session
qracer repl
```

## Architecture

```text
CLI (REPL)
    ↓
ConversationEngine
    ↓
IntentParser ──→ LivePipeline (< 5s)    ← QuickPath
    │               or
    └──────────→ ResearchPipeline        ← DeepPath
                    │
                    ├─ 1. Universe Screening     (Researcher)
                    ├─ 2. Macro Regime Detection  (Analyst)
                    ├─ 3. Cross-Market Discovery  (Analyst)
                    ├─ 4. Consensus Mapping       (Researcher)
                    ├─ 5. Contrarian Detection    (Strategist)
                    ├─ 6. Conviction Scoring      (Strategist)
                    ├─ 7. Trade Thesis Generation (Strategist)
                    ├─ 8. Risk Check              (Strategist)
                    └─ 9. Alpha Report            (Reporter)
```

Data providers and LLM providers register capabilities via a **Registry pattern** — agents request by capability, not by source.

## Project Structure

```text
src/tracer/
├── agents/        # LLM agent roles (researcher, analyst, strategist, reporter)
├── config/        # .qracer/ config loading and models
├── conversation/  # Intent parsing, context tracking, engine orchestration
├── data/          # Data provider protocols and adapters (yfinance, etc.)
├── llm/           # LLM provider protocols and adapters (Claude, etc.)
├── memory/        # Session logging, compaction, search
├── models/        # Domain models (Signal, Report, TradeThesis, ToolResult)
├── risk/          # Portfolio risk calculator, position sizing
├── storage/       # DuckDB persistence layer
└── tools/         # Pipeline tool wrappers
```

## Configuration

Settings live in `.qracer/` (project-local or `~/.qracer/`):

```text
.qracer/
├── config.toml        # Global settings (default mode, LLM preferences)
├── providers.toml     # Data source config (enabled, priority, tier)
├── portfolio.toml     # Watchlist, holdings, risk limits
└── credentials.env    # API keys (user-level only, gitignored)
```

## Development

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run pyright

# Test with coverage (80% minimum)
uv run pytest --cov=src/tracer --cov-report=term-missing --cov-fail-under=80
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12+ |
| Package manager | uv |
| Lint / Format | ruff |
| Type checker | pyright |
| Test | pytest + pytest-asyncio |
| LLM | Multi-provider (Claude, OpenAI, Gemini) |
| Data | Multi-source (yfinance, Finnhub, FRED, FMP) |
| Storage | DuckDB (single-file, append-only) |

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | Dual-mode design, provider plugin system, storage |
| [Pipeline](docs/pipeline.md) | 9-step research pipeline and live pipeline |
| [Conversational Layer](docs/conversational-layer.md) | Intent routing, context management, response formats |
| [Risk System](docs/risk-system.md) | Portfolio model, position sizing, exposure limits |
| [Memory System](docs/memory-system.md) | 3-tier session memory architecture |
| [User Experience](docs/user-experience.md) | Natural language interface, dashboard design |
