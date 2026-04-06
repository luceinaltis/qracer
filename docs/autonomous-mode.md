# Autonomous Mode

> **구현 예정** — 자율 모니터링 모드는 아직 구현되지 않았습니다.

qracer runs autonomously during market hours — monitoring watchlists, detecting significant events, and proactively alerting users without being asked.

Users don't configure modes. The agent decides when to act.

## How It Works

```text
Market opens (9:00 KST / 9:30 EST)
    ↓
Autonomous loop starts
    ├── Load watchlist from USER.md
    ├── Subscribe to price/news feeds for watchlist tickers
    └── Monitor for trigger conditions every 1–5 minutes
            ↓
        Trigger detected (price move, news event, etc.)
            ↓
        Evaluate significance (is this worth interrupting the user?)
            ↓
        Run LivePipeline analysis (< 5s)
            ↓
        Push alert to user with context
```

The loop pauses outside market hours and resumes when markets open.

## Trigger Conditions

| Trigger | Threshold | Action |
|---|---|---|
| Price move | > 2% from session open | Quick analysis + alert |
| Volume spike | > 2× 30-day average | Note in next alert |
| Breaking news | High-relevance headline on watchlist ticker | Summarize + alert |
| User alert | User-defined price threshold crossed | Alert immediately |
| Portfolio move | Position P&L moves > 3% | Portfolio update |
| Cross-market signal | Leading indicator fires | Flag for deeper analysis |

## Alert Decision

Not every trigger becomes an alert. Before pushing:

1. **Cooldown check** — was an alert sent for this ticker in the last 30 minutes?
2. **Significance check** — is this materially new vs. what was already discussed today?
3. **Context check** — is this relevant to the current conversation or user's portfolio?

If any check fails, the event is logged silently and included in the next summary.

## Alert Format

Alerts are short and actionable. They tell the user what happened and what it might mean:

```text
⚡ AAPL -3.2% | Volume 2.1× average

Reuters: iPhone tariff risk re-emerges after trade talks stall.
Insider selling picked up last week (3 executives, $12M combined).

Quick take: Sell-off looks news-driven, not structural. Support ~$172.
Worth watching for follow-through tomorrow.
```

Not a report. Not a wall of text. A message from a knowledgeable friend.

## Watchlist Management

The watchlist lives in `USER.md`. The agent monitors all tickers listed there, plus any tickers mentioned in the current session.

```markdown
# USER.md

## Watchlist
- AAPL
- TSMC
- NVDA
- 005930.KS  # Samsung

## Portfolio
- AAPL: 50 shares (long), avg $165
- TSLA: -10 shares (short), avg $210
```

The agent reads this at session start. Changes take effect next session.

## Scheduled Deep Analysis

Beyond reactive alerts, the agent runs scheduled analysis without being asked:

| Schedule | Action |
|---|---|
| Market open (+30min) | Overnight news summary for watchlist |
| Midday | Cross-market signal scan |
| Market close | Day summary: notable moves, open questions |
| Weekly (Sunday) | Watchlist conviction review |

These run as background tasks and are delivered as conversational messages, not formal reports.

## What the User Sees

Users never see "AutonomousMode" or "LivePipeline". They just get messages:

- **Mid-conversation alerts** — inserted between turns when something important happens
- **Session-start briefing** — "Here's what happened overnight / since we last spoke"
- **Proactive questions** — "Samsung dropped 4% this morning on memory price data. Do you want me to dig into the AAPL supply chain implications?"

The agent acts like a colleague who's been watching the markets while you were away.
