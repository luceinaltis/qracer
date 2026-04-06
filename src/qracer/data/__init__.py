from qracer.data.providers import (
    OHLCV,
    AlternativeProvider,
    AlternativeRecord,
    FundamentalData,
    FundamentalProvider,
    MacroIndicator,
    MacroProvider,
    NewsArticle,
    NewsProvider,
    PriceProvider,
)
from qracer.data.registry import DataRegistry
from qracer.data.yfinance_adapter import YfinanceAdapter

__all__ = [
    "OHLCV",
    "AlternativeProvider",
    "AlternativeRecord",
    "DataRegistry",
    "FundamentalData",
    "FundamentalProvider",
    "MacroIndicator",
    "MacroProvider",
    "NewsArticle",
    "NewsProvider",
    "PriceProvider",
    "YfinanceAdapter",
]

# Optional adapter — only available when fredapi is installed.
try:
    from qracer.data.fred_adapter import FredAdapter

    __all__ += ["FredAdapter"]
except ImportError:
    pass
