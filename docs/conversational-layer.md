# Conversational Layer

## Overview

```text
CLI (REPL)
    ↓
ConversationEngine          — context window, mode selection
    ↓
IntentRouter                — classify → LivePipeline or ResearchPipeline
    ├── LivePipeline        — 1-2 tools → immediate response (< 5s)
    └── ResearchPipeline    — 7-step async → notify on completion
    ↓
ResponseFormatter           — format based on pipeline and intent
    ↓
SessionManager              — persist turn, trigger compaction
```

## Conversation Context Window

The ConversationEngine maintains a rolling context window that tracks:

| Context | TTL | Example |
|---|---|---|
| **Active tickers** | Session lifetime | User mentioned AAPL, TSMC, NVDA |
| **Portfolio** | Persistent | User holds AAPL, short TSLA |
| **Discussion topic** | 10 turns | Comparing semiconductor stocks |
| **Macro backdrop** | Session lifetime | Last macro regime result |
| **Recent data** | 5 minutes | Cached prices, news already fetched |

Context is stored in-memory as a structured object, not as raw conversation history. This keeps token usage low while preserving relevant state.

```python
@dataclass
class ConversationContext:
    active_tickers: list[str]        # mentioned in last N turns
    portfolio: dict[str, Position]   # persisted across sessions
    topic: str | None                # current discussion theme
    macro_regime: str | None         # last detected regime
    recent_data: dict[str, ToolResult]  # cached tool results
    turn_history: list[Turn]         # last 20 turns (summarized beyond that)
```

## Follow-Up Resolution

When a query lacks explicit context, the engine resolves it from the context window:

| Query | Context | Resolution |
|---|---|---|
| "What about Samsung?" | Was discussing AAPL semi supply chain | Samsung as AAPL supplier — fetch Samsung data, frame in supply chain context |
| "And the insider trades?" | Last query was about TSMC | Fetch insider trades for TSMC |
| "Compare it to the competitor" | Active ticker is NVDA | Identify top competitor (AMD), run comparison |
| "How's my portfolio doing?" | Portfolio context loaded | Fetch prices for all portfolio tickers |

Resolution rules:
1. Check if query references an active ticker by name or alias
2. Check if query references a tool/data type from a previous turn
3. Check if query is a continuation of the current topic
4. If ambiguous, ask the user — don't guess

## Proactive Alerts

During market hours, Tracer monitors for events relevant to the conversation context and pushes alerts:

### Alert Types

| Type | Trigger | Example |
|---|---|---|
| **Price move** | Active ticker moves > 2% in session | "AAPL just dropped 3.2% — news: iPhone tariff concerns" |
| **News event** | Breaking news on active ticker | "Reuters: TSMC Q1 revenue beats estimates by 8%" |
| **Threshold** | User-set price alert triggered | "NVDA crossed below $800 (your alert)" |
| **Portfolio** | Portfolio position moves significantly | "Your TSLA short is up 5% today" |

### Implementation

- WebSocket feed filtered to active tickers + portfolio tickers
- Alerts are non-blocking — displayed between user turns, not interrupting input
- Alert history kept in session for follow-up ("tell me more about that AAPL drop")
- Rate-limited: max 1 alert per ticker per 5 minutes to avoid noise

## Response Formats

### Quick Answer (LivePipeline)

For price checks, simple lookups, and factual queries:

```text
AAPL: $178.52 (+1.3%) | Vol: 52M | Day range: 176.20–179.10
```

```text
AAPL vs MSFT:
  P/E:  28.5 vs 35.2
  Div:  0.52% vs 0.74%
  YTD:  +12.3% vs +8.7%
```

### Summary (LivePipeline with LLM)

For opinion queries and news summaries:

```text
[AAPL — Quick Take]
Trading at $178.52 (+1.3%). Three news items today: tariff concerns
(negative), services revenue beat (positive), Buffett trimmed position
(neutral). Net sentiment: slightly negative despite price action.
Insider buying picked up last week — worth watching.
```

### Full Analysis (ResearchPipeline)

```text
[ANALYSIS: {ticker/theme} — {date}]
Conviction: {score}/10

WHAT'S HAPPENING
{1-2 sentence direct answer}

EVIDENCE CHAIN
1. {evidence} — Source: {source}, {date}
   → leads to: {conclusion}

ADVERSARIAL CHECK
- {reason this could be wrong}
- {data staleness or reliability caveat}

VERDICT
{Final judgment with conviction score and key qualifier}
```

## Intent Types

| Intent | Pipeline | Example Query |
|---|---|---|
| `price_check` | Live | "What's AAPL at?" |
| `quick_news` | Live | "Any news on TSLA?" |
| `comparison` | Live | "AAPL vs MSFT P/E" |
| `follow_up` | Live | "What about Samsung?" |
| `opinion` | Live | "Is NVDA overvalued?" |
| `macro_check` | Live | "Where's the 10Y?" |
| `alert_set` | Live | "Alert me if AAPL drops below 170" |
| `event_analysis` | Research | "Why did AAPL spike 5% today?" |
| `deep_dive` | Research | "Full analysis on TSMC" |
| `alpha_hunt` | Research | "Where's the hidden alpha?" |
| `cross_market` | Research | "Korea semi data → US AI stocks?" |

## LLM Role Mapping

All role-to-model assignments overridable via config.

| Role | Default Model | Used By |
|---|---|---|
| router | Rule-based (no LLM) | IntentRouter for common patterns |
| researcher | Claude Haiku | IntentRouter (ambiguous), data summarization |
| analyst | Claude Sonnet | ResearchPipeline analysis steps |
| strategist | Claude Opus | Contrarian detection, conviction scoring |
| reporter | Claude Haiku | Session compaction, report formatting |

Live mode avoids Opus entirely to stay within latency budget.

## Tool Result Contract

All pipeline tools return a `ToolResult` subtype:

```python
@dataclass
class ToolResult:
    tool: str
    success: bool
    data: dict
    source: str
    fetched_at: datetime
    is_stale: bool
    latency_ms: int          # tracked for budget enforcement
    error: str | None
```

## Error Handling

| Failure | Behavior |
|---|---|
| Tool timeout in Live mode | Return partial data with staleness caveat |
| Single tool fails | Exclude from evidence; note as "data unavailable" |
| ≥2 tools fail in Research mode | Exit early; list missing data |
| LLM call fails | Retry once; partial result if still failing |
| Rate limit mid-query | Serve from cache if available |
| WebSocket disconnect | Fall back to REST polling; note in session |
| Alert flood | Rate-limit to 1 per ticker per 5 min |
