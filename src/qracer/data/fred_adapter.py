"""FredAdapter — Macroeconomic indicator data via FRED API.

Provides one capability:
- MacroProvider -> GDP, unemployment, CPI, Fed Funds Rate, 10Y Treasury, VIX
"""

from __future__ import annotations

import asyncio

from qracer.data.providers import MacroIndicator

try:
    from fredapi import Fred  # pyright: ignore[reportMissingImports]

    _HAS_FRED = True
except ImportError:
    _HAS_FRED = False

# Mapping of common indicator names to FRED series IDs and units.
_SERIES_MAP: dict[str, tuple[str, str]] = {
    "fed_funds_rate": ("FEDFUNDS", "%"),
    "fedfunds": ("FEDFUNDS", "%"),
    "10y_treasury": ("DGS10", "%"),
    "treasury_10y": ("DGS10", "%"),
    "dgs10": ("DGS10", "%"),
    "cpi": ("CPIAUCSL", "index"),
    "cpiaucsl": ("CPIAUCSL", "index"),
    "gdp": ("GDP", "billions USD"),
    "unemployment": ("UNRATE", "%"),
    "unrate": ("UNRATE", "%"),
    "vix": ("VIXCLS", "index"),
    "vixcls": ("VIXCLS", "index"),
}


def _resolve_series(name: str) -> tuple[str, str]:
    """Resolve a human-friendly indicator name to (series_id, unit).

    Falls back to using *name* as a raw FRED series ID.
    """
    key = name.lower().strip().replace(" ", "_")
    if key in _SERIES_MAP:
        return _SERIES_MAP[key]
    # Treat as raw FRED series ID.
    return (name.upper(), "")


def _fetch_indicator(client: object, name: str) -> MacroIndicator:
    """Synchronous helper — runs in a thread via asyncio.to_thread."""
    series_id, unit = _resolve_series(name)
    series = client.get_series(series_id)  # type: ignore[union-attr]

    # Drop NaN values and get the most recent observation.
    series = series.dropna()
    if series.empty:
        raise ValueError(f"No data available for FRED series '{series_id}'")

    latest_value = float(series.iloc[-1])
    latest_date = series.index[-1].date()

    return MacroIndicator(
        name=series_id,
        value=latest_value,
        date=latest_date,
        source="fred",
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
            raise ValueError(
                "FRED_API_KEY is required. Get one at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        self._client = Fred(api_key=api_key)

    async def get_indicator(self, name: str) -> MacroIndicator:
        """Get the latest value of a macroeconomic indicator."""
        return await asyncio.to_thread(_fetch_indicator, self._client, name)
