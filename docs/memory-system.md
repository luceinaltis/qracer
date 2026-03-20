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

When a session exceeds 8,000 tokens (measured via `tiktoken`), SessionManager compacts the conversation into a Markdown summary using the reporter role (Haiku). The summary replaces raw turns in the active context window; the JSONL is preserved.

## Tier 3 — Search Index (DuckDB)

DuckDB indexes all Tier 2 Markdown summaries for hybrid retrieval: keyword (FTS) + vector similarity (VSS/HNSW). Writes occur only at compaction time.

- Embedding model: `text-embedding-3-small` (OpenAI API) or `all-MiniLM-L6-v2` (local fallback). Configured in `TOOLS.md`.
- Tables: `session_index` (FTS), `session_embeddings` (HNSW).
- Source of truth is the Markdown files; DuckDB is the index only.

The agent calls `memory_search` autonomously when past context may be relevant.

## MEMORY.md vs. Tier 2

- **Tier 2**: auto-generated, per-session. Temporary working memory.
- **MEMORY.md**: cross-session long-term memory. Manually curated or auto-aggregated. Contains active theses and strong multi-session signals. Loaded at session start via `BOOTSTRAP.md`.
