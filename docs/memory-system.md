# Memory System

Session memory uses a three-tier architecture:

## Tier 1 — Raw Audit Log (JSONL)

Append-only per-session log. Every turn: user message, tool calls, tool results, agent response.

```jsonl
{"turn": 1, "role": "user", "content": "Why did AAPL spike?", "ts": "..."}
{"turn": 1, "role": "tool_call", "tool": "fetch_news", "args": {"ticker": "AAPL"}, "ts": "..."}
{"turn": 1, "role": "tool_result", "success": true, "source": "Finnhub", "ts": "..."}
{"turn": 1, "role": "assistant", "content": "...", "conviction": 8, "ts": "..."}
```

Complete audit trail — reconstructs exactly what the agent did and why.

## Tier 2 — Compressed Summary (Markdown)

When a session exceeds 8,000 tokens (rough `len // 4` estimate of the JSONL log), the `ConversationEngine` invokes `SessionCompactor.compact_and_save()` after each turn via `_maybe_compact()`. The reporter role (Haiku) condenses the turns into a concise Markdown summary, which is written to `~/.qracer/summaries/<session_id>.md`. The raw JSONL log is preserved untouched.

## Tier 3 — Search Index (DuckDB)

`MemorySearcher` indexes Tier 2 Markdown summaries in DuckDB for hybrid retrieval: keyword (BM25 via FTS) and, when an embedding function is supplied, vector similarity via DuckDB's `list_cosine_similarity`. The two branches are fused with reciprocal rank fusion so scores from different scales can be combined without normalisation.

- Embedding is pluggable via the `embedding_fn: Callable[[str], list[float]]` parameter — callers can back it with the Claude API, `text-embedding-3-small`, `sentence-transformers`, or any other model. When `embedding_fn` is `None` the searcher falls back to keyword-only search.
- Tables: `session_index` (FTS) and `session_embeddings` (cosine similarity).
- Source of truth is the Markdown files; DuckDB is the index only.

The agent calls `memory_search` autonomously when past context may be relevant.

## Cross-Session Loading

On `qracer repl` startup, the CLI instantiates a file-backed `MemorySearcher` at `~/.qracer/memory_index.duckdb` and re-indexes every Markdown file in `~/.qracer/summaries/`. The number of loaded contexts is printed to the user so returning sessions immediately know how much prior memory is in scope.

## MEMORY.md vs. Tier 2

- **Tier 2**: auto-generated, per-session. Temporary working memory.
- **MEMORY.md**: cross-session long-term memory stored at `~/.qracer/MEMORY.md`. A machine-managed auto region (delimited by `<!-- BEGIN:auto -->` / `<!-- END:auto -->`) holds open theses and upcoming catalysts regenerated from `FactStore` after every thesis save; everything outside the auto region is user-curated free text and preserved verbatim across refreshes.
- **BOOTSTRAP.md**: optional user-authored system prompt extension at `~/.qracer/BOOTSTRAP.md`. Loaded once at `ConversationEngine` init and injected as a `system` turn so preferences ("I'm a long-term value investor") reach the synthesizer without code changes.

### MEMORY.md format

```markdown
# qracer MEMORY.md

*Last updated: 2026-04-15T12:34:56+00:00*

<!-- BEGIN:auto -->
## Active Theses

- **AAPL** (conviction 8/10): Long AAPL on AI tailwinds. Entry $175.00-$180.00, target $200.00, stop $165.00. Catalyst: AI revenue growth (Q2 2026).

## Upcoming Catalysts

- AAPL: AI revenue growth — Q2 2026

<!-- END:auto -->

## Watchpoints

_(User-editable. Anything outside the auto block is preserved across refreshes.)_

## User Preferences

- Risk tolerance:
- Preferred sectors:
```

### CLI commands

- `memory show` — print the current MEMORY.md.
- `memory refresh` — regenerate the auto region from `FactStore` (also happens automatically after each thesis save).
- `memory edit` — open MEMORY.md in `$EDITOR` for hand-curation; the file is seeded on first use.
