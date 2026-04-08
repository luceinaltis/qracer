"""QuickPath formatter — template-based responses without LLM calls.

Produces compact, terminal-friendly output for simple queries like
price checks and news lookups.  Target: < 5 seconds end-to-end.
"""

from __future__ import annotations

from datetime import datetime

from qracer.conversation.intent import Intent, IntentType
from qracer.models import ToolResult
from qracer.risk.models import PortfolioSnapshot

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
        "ja": "  保有銘柄が設定されていません。\n  ~/.qracer/portfolio.tomlに銘柄を追加してください",
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
