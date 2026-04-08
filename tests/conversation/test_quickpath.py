"""Tests for QuickPath template formatting."""

from __future__ import annotations

from qracer.conversation.intent import Intent, IntentType
from qracer.conversation.quickpath import format_quickpath
from qracer.models import ToolResult


def _price_result(ticker: str = "AAPL", price: float = 178.52) -> ToolResult:
    return ToolResult(
        tool="price_event",
        success=True,
        data={
            "ticker": ticker,
            "current_price": price,
            "bars": 1,
            "ohlcv": [
                {
                    "date": "2026-04-06",
                    "open": 176.20,
                    "high": 179.10,
                    "low": 175.50,
                    "close": 178.52,
                    "volume": 52_000_000,
                }
            ],
        },
        source="PriceProvider",
    )


def _news_result(ticker: str = "AAPL") -> ToolResult:
    return ToolResult(
        tool="news",
        success=True,
        data={
            "ticker": ticker,
            "count": 2,
            "articles": [
                {
                    "title": "AAPL beats earnings estimates",
                    "source": "Reuters",
                    "sentiment": 0.8,
                },
                {
                    "title": "iPhone sales slow in China",
                    "source": "Bloomberg",
                    "sentiment": -0.5,
                },
            ],
        },
        source="NewsProvider",
    )


class TestPriceCheck:
    def test_basic_price(self) -> None:
        intent = Intent(IntentType.PRICE_CHECK, tickers=["AAPL"], raw_query="What's AAPL at?")
        text = format_quickpath(intent, [_price_result()])
        assert "AAPL" in text
        assert "$178.52" in text

    def test_includes_change_and_volume(self) -> None:
        intent = Intent(IntentType.PRICE_CHECK, tickers=["AAPL"], raw_query="AAPL price")
        text = format_quickpath(intent, [_price_result()])
        assert "%" in text
        assert "Vol:" in text
        assert "Range:" in text

    def test_price_unavailable(self) -> None:
        intent = Intent(IntentType.PRICE_CHECK, tickers=["AAPL"], raw_query="AAPL price")
        failed = ToolResult(
            tool="price_event", success=False, data={}, source="test", error="no provider"
        )
        text = format_quickpath(intent, [failed])
        assert "unavailable" in text


class TestQuickNews:
    def test_basic_news(self) -> None:
        intent = Intent(IntentType.QUICK_NEWS, tickers=["AAPL"], raw_query="News on AAPL")
        text = format_quickpath(intent, [_news_result()])
        assert "AAPL" in text
        assert "beats earnings" in text
        assert "Reuters" in text

    def test_sentiment_indicators(self) -> None:
        intent = Intent(IntentType.QUICK_NEWS, tickers=["AAPL"], raw_query="News AAPL")
        text = format_quickpath(intent, [_news_result()])
        assert "[+]" in text  # positive sentiment
        assert "[-]" in text  # negative sentiment

    def test_no_news(self) -> None:
        intent = Intent(IntentType.QUICK_NEWS, tickers=["AAPL"], raw_query="News AAPL")
        failed = ToolResult(tool="news", success=False, data={}, source="test", error="no provider")
        text = format_quickpath(intent, [failed])
        assert "no news" in text


# ---------------------------------------------------------------------------
# i18n tests
# ---------------------------------------------------------------------------


class TestQuickPathI18n:
    """Verify quickpath templates render in non-English languages."""

    def test_price_check_korean(self) -> None:
        intent = Intent(IntentType.PRICE_CHECK, tickers=["AAPL"], raw_query="AAPL 가격")
        text = format_quickpath(intent, [_price_result()], language="ko")
        assert "AAPL" in text
        assert "$178.52" in text

    def test_price_unavailable_korean(self) -> None:
        intent = Intent(IntentType.PRICE_CHECK, tickers=["AAPL"], raw_query="AAPL 가격")
        failed = ToolResult(
            tool="price_event", success=False, data={}, source="test", error="no provider"
        )
        text = format_quickpath(intent, [failed], language="ko")
        assert "가격 정보 없음" in text

    def test_price_unavailable_japanese(self) -> None:
        intent = Intent(IntentType.PRICE_CHECK, tickers=["TSLA"], raw_query="TSLA price")
        failed = ToolResult(
            tool="price_event", success=False, data={}, source="test", error="no provider"
        )
        text = format_quickpath(intent, [failed], language="ja")
        assert "価格情報なし" in text

    def test_news_header_korean(self) -> None:
        intent = Intent(IntentType.QUICK_NEWS, tickers=["AAPL"], raw_query="AAPL 뉴스")
        text = format_quickpath(intent, [_news_result()], language="ko")
        assert "뉴스" in text
        assert "2건" in text

    def test_no_news_japanese(self) -> None:
        intent = Intent(IntentType.QUICK_NEWS, tickers=["AAPL"], raw_query="AAPL news")
        failed = ToolResult(tool="news", success=False, data={}, source="test", error="no provider")
        text = format_quickpath(intent, [failed], language="ja")
        assert "ニュースなし" in text

    def test_no_data_korean(self) -> None:
        intent = Intent(IntentType.ALPHA_HUNT, tickers=["AAPL"], raw_query="test")
        text = format_quickpath(intent, [], language="ko")
        assert "데이터 없음" in text

    def test_english_default_preserved(self) -> None:
        """Default language=en should produce identical output to no language arg."""
        intent = Intent(IntentType.PRICE_CHECK, tickers=["AAPL"], raw_query="AAPL price")
        failed = ToolResult(
            tool="price_event", success=False, data={}, source="test", error="no provider"
        )
        text_default = format_quickpath(intent, [failed])
        text_en = format_quickpath(intent, [failed], language="en")
        assert text_default == text_en
        assert "unavailable" in text_en


class TestKeywordDetection:
    def test_price_check_keywords(self) -> None:
        from unittest.mock import AsyncMock

        from qracer.conversation.intent import IntentParser, IntentType
        from qracer.llm.providers import Role
        from qracer.llm.registry import LLMRegistry

        # Use failing LLM to trigger keyword fallback
        mock = AsyncMock()
        mock.complete.side_effect = RuntimeError("fail")
        registry = LLMRegistry()
        registry.register("mock", mock, [Role.RESEARCHER])
        parser = IntentParser(registry)

        import asyncio

        intent = asyncio.get_event_loop().run_until_complete(parser.parse("What's AAPL at?"))
        assert intent.intent_type == IntentType.PRICE_CHECK

    def test_quick_news_keywords(self) -> None:
        from unittest.mock import AsyncMock

        from qracer.conversation.intent import IntentParser, IntentType
        from qracer.llm.providers import Role
        from qracer.llm.registry import LLMRegistry

        mock = AsyncMock()
        mock.complete.side_effect = RuntimeError("fail")
        registry = LLMRegistry()
        registry.register("mock", mock, [Role.RESEARCHER])
        parser = IntentParser(registry)

        import asyncio

        intent = asyncio.get_event_loop().run_until_complete(parser.parse("Any news on TSLA?"))
        assert intent.intent_type == IntentType.QUICK_NEWS
