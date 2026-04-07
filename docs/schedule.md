# Schedule — Task Scheduler

General-purpose task scheduler for qracer. Records tasks to execute later — both one-time and recurring — and runs them when due.

## Task Types

| Type | Example | Description |
|------|---------|-------------|
| `analyze` | `schedule analyze AAPL every 1h` | Run full analysis on a ticker |
| `news_scan` | `schedule news scan TSLA every 30m` | Fetch latest news articles |
| `portfolio_snapshot` | `schedule portfolio snapshot daily 09:30` | Take a portfolio P&L snapshot |
| `cross_market_scan` | `schedule cross market AAPL,TSLA every 1d` | Cross-market comparison |
| `custom_query` | `schedule query "macro outlook" every 1d` | Freeform query via engine |

## Schedule Formats

| Format | Example | Type |
|--------|---------|------|
| ISO datetime | `2026-04-08T09:30:00` | One-time |
| Interval | `every 1h`, `every 30m`, `every 1d` | Recurring |
| Daily | `daily 09:30` | Recurring |
| Weekly | `weekly monday 09:00` | Recurring |

## Execution

When the REPL is running, due tasks are checked every 30 seconds before each prompt. Results are displayed inline.

## REPL Commands

| Command | Description |
|---------|-------------|
| `schedule <action> every/at <time>` | Create a task |
| `tasks` | List all tasks |
| `cancel-task <id>` | Cancel a task |

## Storage

Tasks are persisted in `~/.qracer/tasks.json`. Each task tracks:
- Action type and parameters
- Schedule spec and next run time
- Status (pending/running/completed/failed)
- Run count and last error
