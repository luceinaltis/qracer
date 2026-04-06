"""YfinanceAdapter — Price/OHLCV data via the yfinance library.

Used as a fallback for price data. Only provides Price/OHLCV capability.
yfinance is unofficial and can get IP-blocked; use only for historical data backfill.
"""

from __future__ import annotations

import asyncio
from datetime import date

from qracer.data.providers import OHLCV, PriceProvider

try:
    import yfinance as yf

    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False


def _fetch_price(ticker: str) -> float:
    """Synchronous helper — runs in a thread via asyncio.to_thread."""
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1d")
    if hist.empty:
        raise ValueError(f"No price data available for {ticker}")
    return float(hist["Close"].iloc[-1])


def _fetch_ohlcv(ticker: str, start: date, end: date) -> list[OHLCV]:
    """Synchronous helper — runs in a thread via asyncio.to_thread."""
    tk = yf.Ticker(ticker)
    hist = tk.history(start=start.isoformat(), end=end.isoformat())
    if hist.empty:
        return []
    bars: list[OHLCV] = []
    from typing import cast

    import pandas as pd

    for idx, row in hist.iterrows():
        ts = cast(pd.Timestamp, idx)
        o = cast(float, row["Open"])
        h = cast(float, row["High"])
        lo = cast(float, row["Low"])
        c = cast(float, row["Close"])
        v = cast(int, row["Volume"])
        bars.append(
            OHLCV(
                date=ts.date(),
                open=float(o),
                high=float(h),
                low=float(lo),
                close=float(c),
                volume=int(v),
            )
        )
    return bars


class YfinanceAdapter:
    """Data adapter for yfinance (Price/OHLCV only)."""

    capabilities: list = [PriceProvider]

    def __init__(self, api_key: str | None = None) -> None:
        if not _HAS_YFINANCE:
            raise ImportError("yfinance is not installed. Install it with: uv add yfinance")

    async def get_price(self, ticker: str) -> float:
        """Get the latest closing price for a ticker."""
        return await asyncio.to_thread(_fetch_price, ticker)

    async def get_ohlcv(self, ticker: str, start: date, end: date) -> list[OHLCV]:
        """Get OHLCV bars for a date range."""
        return await asyncio.to_thread(_fetch_ohlcv, ticker, start, end)
