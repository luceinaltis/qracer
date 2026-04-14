"""QuickPath formatter — template-based responses without LLM calls.

Produces compact, terminal-friendly output for simple queries like
price checks and news lookups.  Target: < 5 seconds end-to-end.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from qracer.alerts import Alert, AlertStore
from qracer.conversation.intent import Intent, IntentType
from qracer.data.providers import PriceProvider
from qracer.data.registry import DataRegistry
from qracer.memory.fact_store import FactStore
from qracer.models import ToolResult
from qracer.risk.models import PortfolioSnapshot
from qracer.tasks import TaskStore
from qracer.watchlist import Watchlist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# i18n template lookup tables
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, dict[str, str]] = {
    "price_unavailable": {
        "en": "{ticker}: price unavailable",
        "ko": "{ticker}: 가격 정보 없음",
        "ja": "{ticker}: 価格情報なし",
    },
    "no_news_available": {
        "en": "{ticker}: no news available",
        "ko": "{ticker}: 뉴스 없음",
        "ja": "{ticker}: ニュースなし",
    },
    "no_recent_news": {
        "en": "{ticker}: no recent news",
        "ko": "{ticker}: 최근 뉴스 없음",
        "ja": "{ticker}: 最近のニュースなし",
    },
    "news_header": {
        "en": "News for {ticker} ({count} articles):",
        "ko": "{ticker} 뉴스 ({count}건):",
        "ja": "{ticker} ニュース ({count}件):",
    },
    "no_data": {
        "en": "No data available.",
        "ko": "데이터 없음.",
        "ja": "データなし。",
    },
    "portfolio_summary": {
        "en": "Portfolio Summary (as of {time})",
        "ko": "포트폴리오 요약 ({time} 기준)",
        "ja": "ポートフォリオ概要（{time}時点）",
    },
    "no_holdings": {
        "en": "  No holdings configured.\n  Add holdings to ~/.qracer/portfolio.toml",
        "ko": "  보유 종목이 없습니다.\n  ~/.qracer/portfolio.toml에 종목을 추가하세요",
        "ja": "  保有銘柄が設定されていません。\n"
        "  ~/.qracer/portfolio.tomlに銘柄を追加してください",
    },
}


def _t(key: str, language: str = "en", **kwargs: object) -> str:
    """Look up a translated template and format it with kwargs."""
    templates = _TEMPLATES.get(key, {})
    template = templates.get(language, templates.get("en", key))
    return template.format(**kwargs)


def format_quickpath(intent: Intent, results: list[ToolResult], *, language: str = "en") -> str:
    """Format tool results into a compact template response.

    No LLM call required — pure string formatting.
    """
    if intent.intent_type == IntentType.PRICE_CHECK:
        return _format_price_check(intent, results, language=language)
    if intent.intent_type == IntentType.QUICK_NEWS:
        return _format_quick_news(intent, results, language=language)
    return _format_generic(intent, results, language=language)


def format_portfolio(snapshot: PortfolioSnapshot, *, language: str = "en") -> str:
    """Format a portfolio snapshot into a readable summary table."""
    now = datetime.now().strftime("%H:%M")
    lines = [_t("portfolio_summary", language, time=now)]
    lines.append("")

    if not snapshot.holdings:
        lines.append(_t("no_holdings", language))
        return "\n".join(lines)

    # Header
    lines.append(f"  {'Ticker':<8}{'Shares':>8}{'Price':>12}{'Value':>14}{'P&L':>14}{'%':>8}")
    lines.append("  " + "─" * 64)

    total_pnl = 0.0
    for h in snapshot.holdings:
        sign = "+" if h.unrealized_pnl >= 0 else ""
        lines.append(
            f"  {h.ticker:<8}"
            f"{h.shares:>8.0f}"
            f"  ${h.current_price:>9,.2f}"
            f"  ${h.market_value:>11,.2f}"
            f"  {sign}${abs(h.unrealized_pnl):>10,.2f}"
            f"  {sign}{h.unrealized_pnl_pct:.1f}%"
        )
        total_pnl += h.unrealized_pnl

    lines.append("  " + "─" * 64)
    sign = "+" if total_pnl >= 0 else ""
    lines.append(f"  Total: ${snapshot.total_value:,.2f}  |  P&L: {sign}${abs(total_pnl):,.2f}")

    return "\n".join(lines)


def _format_price_check(intent: Intent, results: list[ToolResult], *, language: str = "en") -> str:
    """Format price data into a compact one-liner."""
    ticker = intent.tickers[0] if intent.tickers else "?"
    price_result = next((r for r in results if r.tool == "price_event" and r.success), None)

    if price_result is None:
        return _t("price_unavailable", language, ticker=ticker)

    data = price_result.data
    price = data.get("current_price")
    bars = data.get("ohlcv", [])

    if price is None:
        return _t("price_unavailable", language, ticker=ticker)

    parts = [f"{ticker}: ${price:,.2f}"]

    # Calculate day change from latest OHLCV bar if available.
    if bars:
        latest = bars[-1]
        prev_close = latest.get("open", price)
        if prev_close and prev_close > 0:
            change = price - prev_close
            change_pct = (change / prev_close) * 100
            sign = "+" if change >= 0 else ""
            parts.append(f"{sign}{change_pct:.1f}%")

        # Volume
        vol = latest.get("volume")
        if vol:
            if vol >= 1_000_000:
                parts.append(f"Vol: {vol / 1_000_000:.1f}M")
            elif vol >= 1_000:
                parts.append(f"Vol: {vol / 1_000:.0f}K")

        # Day range
        high = latest.get("high")
        low = latest.get("low")
        if high and low:
            parts.append(f"Range: {low:.2f}–{high:.2f}")

    return " | ".join(parts)


def _format_quick_news(intent: Intent, results: list[ToolResult], *, language: str = "en") -> str:
    """Format news articles into a brief list."""
    ticker = intent.tickers[0] if intent.tickers else "?"
    news_result = next((r for r in results if r.tool == "news" and r.success), None)

    if news_result is None:
        return _t("no_news_available", language, ticker=ticker)

    articles = news_result.data.get("articles", [])
    if not articles:
        return _t("no_recent_news", language, ticker=ticker)

    lines = [_t("news_header", language, ticker=ticker, count=len(articles))]
    for a in articles[:5]:
        title = a.get("title", "Untitled")
        source = a.get("source", "")
        sentiment = a.get("sentiment")
        sentiment_str = ""
        if sentiment is not None:
            if sentiment > 0.3:
                sentiment_str = " [+]"
            elif sentiment < -0.3:
                sentiment_str = " [-]"
        src = f" ({source})" if source else ""
        lines.append(f"  - {title}{src}{sentiment_str}")

    return "\n".join(lines)


def _format_generic(intent: Intent, results: list[ToolResult], *, language: str = "en") -> str:
    """Fallback for unhandled QuickPath intents."""
    successful = [r for r in results if r.success]
    if not successful:
        return _t("no_data", language)
    parts = []
    for r in successful:
        parts.append(f"[{r.tool}] {r.source}: {len(r.data)} fields")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Session-start briefing
# ---------------------------------------------------------------------------


async def generate_briefing(
    watchlist: Watchlist,
    data_registry: DataRegistry,
    alert_store: AlertStore,
    task_store: TaskStore,
    sessions_dir: Path,
    current_session: Path | None = None,
    fact_store: FactStore | None = None,
) -> str | None:
    """Generate a session-start briefing.

    Summarises activity since the previous session: current watchlist
    prices, alerts that triggered while away, and any pending scheduled
    tasks.  Returns ``None`` when there is no previous session on disk
    or when nothing noteworthy is available.

    Args:
        watchlist: User's ticker watchlist.
        data_registry: Provides ``PriceProvider`` for live price lookups.
        alert_store: Source of triggered alerts.
        task_store: Source of pending scheduled tasks.
        sessions_dir: Directory containing session JSONL log files.
        current_session: Path of the current session log; excluded from
            "last session" detection so the briefing reflects activity
            since the previous run, not the current one.
    """
    last_session = _find_last_session(sessions_dir, current_session=current_session)
    if last_session is None:
        return None

    lines: list[str] = ["Session Briefing", "=" * 40, ""]
    last_session_label = last_session.astimezone().strftime("%Y-%m-%d %H:%M")
    lines.append(f"Since last session ({last_session_label}):")
    lines.append("")
    has_content = False

    # Watchlist prices
    price_lines = await _briefing_price_lines(watchlist, data_registry)
    if price_lines:
        lines.append("Watchlist:")
        lines.extend(price_lines)
        lines.append("")
        has_content = True

    # Alerts triggered since the last session
    triggered_lines = _briefing_alert_lines(alert_store, since=last_session)
    if triggered_lines:
        lines.append(f"Triggered Alerts ({len(triggered_lines)}):")
        lines.extend(triggered_lines)
        lines.append("")
        has_content = True

    # Pending tasks
    task_lines, pending_count = _briefing_task_lines(task_store)
    if task_lines:
        lines.append(f"Pending Tasks ({pending_count}):")
        lines.extend(task_lines)
        lines.append("")
        has_content = True

    # Open theses from fact store
    if fact_store is not None:
        thesis_lines = _briefing_thesis_lines(fact_store)
        if thesis_lines:
            lines.append(f"Open Theses ({len(thesis_lines)}):")
            lines.extend(thesis_lines)
            lines.append("")
            has_content = True

    if not has_content:
        return None

    # Drop trailing blank lines.
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _briefing_thesis_lines(fact_store: FactStore) -> list[str]:
    """Format open theses for the session-start briefing."""
    upcoming = fact_store.get_upcoming_catalysts(days_ahead=14)
    if not upcoming:
        upcoming = fact_store.get_open_theses()[:5]
    lines: list[str] = []
    for t in upcoming:
        entry_mid = (t.entry_zone_low + t.entry_zone_high) / 2
        direction = "LONG" if t.target_price > entry_mid else "SHORT"
        line = (
            f"  {direction} {t.ticker}: conviction {t.conviction}/10, target ${t.target_price:.2f}"
        )
        if t.catalyst:
            line += f", catalyst: {t.catalyst}"
            if t.catalyst_date:
                line += f" ({t.catalyst_date})"
        lines.append(line)
    return lines


def _find_last_session(
    sessions_dir: Path,
    current_session: Path | None = None,
) -> datetime | None:
    """Return the mtime of the most recent session log file.

    Excludes ``current_session`` (when provided) so that callers can ask
    "what is the previous session?" even after the current session has
    started writing.
    """
    if not sessions_dir.exists():
        return None

    try:
        files = list(sessions_dir.glob("*.jsonl"))
    except OSError:
        return None

    current_resolved: Path | None = None
    if current_session is not None:
        try:
            current_resolved = current_session.resolve()
        except OSError:
            current_resolved = current_session

    candidates: list[tuple[float, Path]] = []
    for path in files:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if current_resolved is not None and resolved == current_resolved:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return datetime.fromtimestamp(candidates[0][0], tz=timezone.utc)


async def _briefing_price_lines(
    watchlist: Watchlist,
    data_registry: DataRegistry,
) -> list[str]:
    """Fetch current prices for watchlist tickers, skipping failures."""
    lines: list[str] = []
    for ticker in watchlist.tickers:
        try:
            price = await data_registry.async_get_with_fallback(PriceProvider, "get_price", ticker)
        except Exception:
            logger.debug("Briefing price fetch failed for %s", ticker, exc_info=True)
            continue
        if not isinstance(price, (int, float)):
            continue
        lines.append(f"  {ticker}: ${price:,.2f}")
    return lines


def _briefing_alert_lines(alert_store: AlertStore, since: datetime) -> list[str]:
    """Return formatted lines for alerts triggered after ``since``."""
    triggered: list[tuple[datetime, Alert]] = []
    for alert in alert_store.alerts:
        if alert.active or not alert.triggered_at:
            continue
        try:
            dt = datetime.fromisoformat(alert.triggered_at)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt > since:
            triggered.append((dt, alert))

    triggered.sort(key=lambda item: item[0], reverse=True)
    return [
        f"  - {alert.describe()} (at {dt.astimezone().strftime('%Y-%m-%d %H:%M')})"
        for dt, alert in triggered
    ]


def _briefing_task_lines(task_store: TaskStore) -> tuple[list[str], int]:
    """Return formatted lines for the first few pending tasks plus the total count."""
    pending = task_store.get_active()
    if not pending:
        return [], 0
    lines = [f"  [{t.id}] {t.describe()}" for t in pending[:5]]
    if len(pending) > 5:
        lines.append(f"  ... and {len(pending) - 5} more")
    return lines, len(pending)
