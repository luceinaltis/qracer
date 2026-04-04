# HEARTBEAT.md

Autonomous monitoring tasks for Tracer. Runs periodically during market hours.

## Market Hours Check (9:00–15:30 KST / 9:30–16:00 EST)

- Fetch latest prices for all tickers in USER.md watchlist
- Check for price moves > 2% from session open
- Scan for breaking news on watchlist tickers
- If trigger found: run LivePipeline analysis → push alert to user

## Session Start Briefing

At the start of each session, summarize:
- Overnight / since-last-session moves on watchlist
- Key news events
- Any open alerts or pending analysis from last session

## Scheduled Analysis

| Schedule | Task |
|---|---|
| Market open +30min | Overnight news summary |
| Midday | Cross-market signal scan |
| Market close | Day summary for watchlist |
| Weekly (Sunday) | Watchlist conviction review |

## Cooldown Rules

- Max 1 alert per ticker per 30 minutes
- Skip alerts outside market hours (unless user-defined threshold crossed)
- Batch minor events into summary instead of individual alerts
