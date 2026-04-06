"""Built-in provider catalog — maps provider names to adapter classes.

New providers only need an entry here and in providers.toml.
The registry builder uses this catalog for dynamic import and registration.
"""

from __future__ import annotations

# Data providers: name -> (adapter import path, [capability import paths])
BUILTIN_DATA_PROVIDERS: dict[str, tuple[str, list[str]]] = {
    "yfinance": (
        "qracer.data.yfinance_adapter.YfinanceAdapter",
        [
            "qracer.data.providers.PriceProvider",
            "qracer.data.providers.FundamentalProvider",
            "qracer.data.providers.NewsProvider",
        ],
    ),
}

# LLM providers: name -> (adapter import path, [role enum values])
BUILTIN_LLM_PROVIDERS: dict[str, tuple[str, list[str]]] = {
    "claude": (
        "qracer.llm.claude_adapter.ClaudeAdapter",
        [
            "researcher",
            "analyst",
            "strategist",
            "reporter",
        ],
    ),
}
