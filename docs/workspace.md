# Workspace

The qracer workspace is a single directory containing all project files.

## Structure

```text
qracer/
├── src/qracer/       # Source code
├── tests/            # Test suite
├── docs/             # Design documentation
├── skills/           # Agent skill definitions
└── scripts/          # CI validation scripts
```

## Development Setup

```bash
uv sync              # Install dependencies
uv run pytest        # Run tests
uv run ruff check .  # Lint
uv run pyright       # Type check
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
ANTHROPIC_API_KEY=...
FINNHUB_API_KEY=...
FRED_API_KEY=...
```

Never commit `.env`. It is gitignored.
