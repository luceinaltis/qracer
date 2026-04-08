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

> **구현 예정** — MEMORY.md, BOOTSTRAP.md 기반 크로스 세션 메모리는 아직 구현되지 않았습니다.

- **Tier 2**: auto-generated, per-session. Temporary working memory.
- **MEMORY.md**: cross-session long-term memory. Manually curated or auto-aggregated. Contains active theses and strong multi-session signals. Loaded at session start via `BOOTSTRAP.md`.
