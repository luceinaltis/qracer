"""FredAdapter — Macroeconomic indicator data via FRED API.

Provides one capability:
- MacroProvider → macroeconomic indicators (Fed Funds Rate, CPI, GDP, etc.)

Uses the ``fredapi`` package for FRED REST API access.
Rate limit: 120 requests/minute.
"""

from __future__ import annotations

import asyncio
from datetime import date

from qracer.data.providers import MacroIndicator

try:
    from fredapi import Fred  # pyright: ignore[reportMissingImports]

    _HAS_FRED = True
except ImportError:
    _HAS_FRED = False

# Mapping from friendly indicator names to FRED series IDs and units.
_INDICATOR_MAP: dict[str, tuple[str, str]] = {
    "fed_funds_rate": ("FEDFUNDS", "%"),
    "treasury_10y": ("DGS10", "%"),
    "cpi_yoy": ("CPIAUCSL", "index"),
    "gdp_growth": ("GDP", "billions USD"),
    "unemployment": ("UNRATE", "%"),
    "vix": ("VIXCLS", "index"),
}


def _fetch_indicator(client: object, name: str) -> MacroIndicator:
    """Synchronous helper — runs in a thread via asyncio.to_thread."""
    key = name.lower().strip()
    if key in _INDICATOR_MAP:
        series_id, unit = _INDICATOR_MAP[key]
    else:
        # Allow direct FRED series IDs as a fallback.
        series_id = key.upper()
        unit = ""

    series = client.get_series(series_id)  # type: ignore[union-attr]
    if series is None or series.empty:
        msg = f"No data returned for FRED series '{series_id}'"
        raise RuntimeError(msg)

    # Get the most recent non-NaN observation.
    series = series.dropna()
    if series.empty:
        msg = f"No valid observations for FRED series '{series_id}'"
        raise RuntimeError(msg)

    latest_value = float(series.iloc[-1])
    latest_date = series.index[-1]

    return MacroIndicator(
        name=name,
        value=latest_value,
        date=date(latest_date.year, latest_date.month, latest_date.day),
        source="FRED",
        unit=unit,
    )


class FredAdapter:
    """Data adapter for FRED (Macroeconomic indicators)."""

    def __init__(self, api_key: str | None = None) -> None:
        if not _HAS_FRED:
            raise ImportError(
                "fredapi is not installed. Install it with: uv add fredapi"
            )
        if not api_key:
            msg = "FRED API key is required. Set FRED_API_KEY environment variable."
            raise ValueError(msg)
        self._client = Fred(api_key=api_key)

    async def get_indicator(self, name: str) -> MacroIndicator:
        """Fetch the latest value of a macroeconomic indicator."""
        return await asyncio.to_thread(_fetch_indicator, self._client, name)
