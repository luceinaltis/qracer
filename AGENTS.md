# AGENTS.md

## Project Overview

Tracer — conversational investment agent for professional investors.
Discovers alpha in global security markets via cross-market analysis and contrarian signal detection.
Users query in natural language; the agent fetches real data, builds causal reasoning chains, and responds with calibrated conviction.

## Agent Guidelines

Before starting any task, check `.claude/skills/` for a matching skill and follow it.
Skills define workflows for feature dev, PR review, testing, refactoring, and more — always use them over ad-hoc approaches.

## Design Documents

Before implementing any feature or making architectural changes, read the relevant docs first:

| Document | When to read |
|----------|-------------|
| `docs/architecture.md` | Any new feature or structural change |
| `docs/pipeline.md` | Data flow or pipeline modifications |
| `docs/conversational-layer.md` | Conversation handling changes |
| `docs/memory-system.md` | Memory or state management changes |
| `docs/workspace.md` | Workspace configuration changes |

## Tech Stack

- **Language**: Python 3.12+
- **Package manager**: uv
- **Linter/Formatter**: ruff
- **Type checker**: pyright
- **Test**: pytest
- **LLM**: multi-provider (Claude, OpenAI, Gemini)
- **Data**: multi-source (Finnhub, yfinance, FRED, FMP)
- **Storage**: DuckDB (single-file, append-only)
