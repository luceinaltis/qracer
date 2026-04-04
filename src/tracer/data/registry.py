"""DataRegistry — capability-based routing with fallback.

Adapters register capabilities. Agents request data by capability, not by source.
If the primary adapter fails or is unavailable, the registry falls through to fallbacks.
"""

from __future__ import annotations

from typing import Any

from tracer.data.providers import (
    AlternativeProvider,
    FundamentalProvider,
    MacroProvider,
    NewsProvider,
    PriceProvider,
)

# Union of all provider protocols
ProviderType = type[
    PriceProvider | FundamentalProvider | MacroProvider | NewsProvider | AlternativeProvider
]


class DataRegistry:
    """Routes data requests by capability with ordered fallback."""

    def __init__(self) -> None:
        # capability -> list of (name, adapter) in priority order
        self._adapters: dict[ProviderType, list[tuple[str, Any]]] = {}

    def register(self, name: str, adapter: Any, capabilities: list[ProviderType]) -> None:
        """Register an adapter with its capabilities.

        Adapters registered first for a given capability have higher priority.
        """
        for cap in capabilities:
            if cap not in self._adapters:
                self._adapters[cap] = []
            self._adapters[cap].append((name, adapter))

    def get(self, capability: ProviderType, name: str | None = None) -> Any:
        """Get an adapter by capability, optionally by explicit name.

        Args:
            capability: The provider protocol to look up.
            name: If provided, return the adapter with this name specifically.

        Returns:
            The adapter instance.

        Raises:
            KeyError: If no adapter is registered for the capability (or name).
        """
        adapters = self._adapters.get(capability)
        if not adapters:
            raise KeyError(f"No adapter registered for capability {capability.__name__}")

        if name is not None:
            for adapter_name, adapter in adapters:
                if adapter_name == name:
                    return adapter
            raise KeyError(f"No adapter named '{name}' for capability {capability.__name__}")

        # Return the first (highest priority) adapter
        return adapters[0][1]

    def get_all(self, capability: ProviderType) -> list[tuple[str, Any]]:
        """Get all adapters for a capability in priority order."""
        return list(self._adapters.get(capability, []))

    def capabilities(self) -> list[ProviderType]:
        """List all registered capabilities."""
        return list(self._adapters.keys())
