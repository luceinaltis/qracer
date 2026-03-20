# AGENTS.md

## Project Overview

Tracer — conversational investment agent for professional investors.
Discovers alpha in global security markets via cross-market analysis and contrarian signal detection.
Users query in natural language; the agent fetches real data, builds causal reasoning chains, and responds with calibrated conviction.

For detailed design, see `docs/`. For executable workflows, see `skills/`.

## Core Principles

1. **Data-First** — evidence before opinion. Never start from intuition.
2. **No Hallucination** — never fabricate data. If unavailable, say so explicitly.
3. **Adversarial Self-Check** — argue against every conclusion before committing.
4. **Calibrated Conviction** — score signals 1-10, not binary. Weak signals are kept, not discarded.
5. **Autonomous Reasoning** — agent self-queries when data is insufficient (max 3 iterations or cost limit).
6. **Traceable Reasoning** — every judgment has a source → analysis → conclusion chain.
7. **Cost-Aware** — each additional query must clear a value > cost threshold.

## Tech Stack

- **Language**: Python 3.12+
- **Package manager**: uv
- **Linter/Formatter**: ruff
- **Type checker**: pyright
- **Test**: pytest
- **LLM**: multi-provider (Claude, OpenAI, Gemini)
- **Data**: multi-source (Finnhub, yfinance, FRED, FMP)
- **Storage**: DuckDB (single-file, append-only)

## Project Structure

```
tracer/
├── AGENTS.md
├── pyproject.toml
├── src/tracer/
│   ├── agents/            # Agent roles (researcher, analyst, strategist, reporter)
│   ├── conversation/      # Conversational layer (engine, intent, synthesizer, session)
│   ├── memory/            # Session memory search (FTS + HNSW via DuckDB)
│   ├── tools/             # Pipeline tool wrappers (price_event, news, insider, etc.)
│   ├── llm/               # LLM adapter + capability registry
│   ├── data/              # Data adapter + capability registry
│   ├── models/            # Domain models (Stock, Signal, Report, etc.)
│   ├── storage/           # DuckDB persistence layer
│   └── config/            # Configuration loader
├── tests/                 # Mirrors src/ structure
├── docs/                  # Design documentation (architecture, pipeline, memory, etc.)
├── skills/                # Agent skill definitions (SKILL.md per skill)
└── scripts/               # CLI entry points
```

## Coding Conventions

### Absolute Rules

- **English only.** All code, comments, docstrings, commits, docs in English.
- **No hallucination in code.** Never swallow errors silently. Failed API calls surface as explicit errors.
- **Dependency inversion.** Agents depend on protocols, never concrete adapters.
  - `def analyze(provider: PriceProvider)` ✓
  - `def analyze(provider: FinnhubAdapter)` ✗

### Style

- Type hints on all public functions
- `async/await` for all I/O-bound operations
- Google-style docstrings on all public functions/classes
- Comments explain **why**, not what. Keep them minimal.
- snake_case functions/variables, PascalCase classes, UPPER_CASE constants
- Files under 300 LOC; split when exceeded
- DRY: extract at 3+ occurrences, not 2 (avoid premature abstraction)

### Git

- Conventional Commits: `feat|fix|refactor|build|ci|chore|docs|style|perf|test`
- Atomic, focused commits
- Branch naming: `feat/<name>`, `fix/<name>`, `refactor/<name>`
- Never commit secrets. Use `.env` (gitignored).

## Common Commands

```bash
uv sync                              # Install dependencies
ruff check . && ruff format --check . # Lint + format check
pytest                                # Run tests
pyright                               # Type check
```
