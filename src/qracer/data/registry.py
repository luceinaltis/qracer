"""DataRegistry — capability-based routing with fallback.

Adapters register capabilities. Agents request data by capability, not by source.
If the primary adapter fails or is unavailable, the registry falls through to fallbacks.
"""

from __future__ import annotations

import logging
from typing import Any

# Provider protocol classes are used as capability keys (e.g. PriceProvider, NewsProvider).
# pyright doesn't support type[Protocol], so we use type[Any] as the capability type.
ProviderType = type[Any]

logger = logging.getLogger(__name__)


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
        """Get the highest-priority adapter for a capability.

        This is a simple lookup — it returns the adapter instance without
        invoking any methods.  For call-level fallback (try the primary
        adapter, fall through to the next on failure) use
        :meth:`get_with_fallback` or :meth:`async_get_with_fallback`.

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

        return adapters[0][1]

    def get_with_fallback(
        self,
        capability: ProviderType,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call *method* on each adapter for *capability* until one succeeds.

        Args:
            capability: The provider protocol to look up.
            method: The method name to call on the adapter.
            *args: Positional arguments forwarded to the method.
            **kwargs: Keyword arguments forwarded to the method.

        Returns:
            The return value from the first successful adapter call.

        Raises:
            KeyError: If no adapter is registered for the capability.
            Exception: The last exception if all adapters fail.
        """
        adapters = self._adapters.get(capability)
        if not adapters:
            raise KeyError(f"No adapter registered for capability {capability.__name__}")

        last_exc: Exception | None = None
        for adapter_name, adapter in adapters:
            try:
                fn = getattr(adapter, method)
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Adapter '%s'.%s failed, trying next: %s",
                    adapter_name,
                    method,
                    exc,
                )

        raise last_exc  # type: ignore[misc]

    async def async_get_with_fallback(
        self,
        capability: ProviderType,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call an async *method* on each adapter for *capability* until one succeeds.

        Same as :meth:`get_with_fallback` but awaits the adapter method,
        making it suitable for the async pipeline tools.

        Args:
            capability: The provider protocol to look up.
            method: The async method name to call on the adapter.
            *args: Positional arguments forwarded to the method.
            **kwargs: Keyword arguments forwarded to the method.

        Returns:
            The return value from the first successful adapter call.

        Raises:
            KeyError: If no adapter is registered for the capability.
            Exception: The last exception if all adapters fail.
        """
        adapters = self._adapters.get(capability)
        if not adapters:
            raise KeyError(f"No adapter registered for capability {capability.__name__}")

        last_exc: Exception | None = None
        for adapter_name, adapter in adapters:
            try:
                fn = getattr(adapter, method)
                return await fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Adapter '%s'.%s failed, trying next: %s",
                    adapter_name,
                    method,
                    exc,
                )

        raise last_exc  # type: ignore[misc]

    def get_all(self, capability: ProviderType) -> list[tuple[str, Any]]:
        """Get all adapters for a capability in priority order."""
        return list(self._adapters.get(capability, []))

    def capabilities(self) -> list[ProviderType]:
        """List all registered capabilities."""
        return list(self._adapters.keys())
