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

# Built-in provider name → (adapter class import path, capabilities list import path)
_BUILTIN_PROVIDERS: dict[str, tuple[str, list[str]]] = {
    "yfinance": (
        "qracer.data.yfinance_adapter.YfinanceAdapter",
        ["qracer.data.providers.PriceProvider"],
    ),
}


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
        """Get an adapter by capability with fallback, optionally by explicit name.

        When *name* is None the method tries each registered adapter in priority
        order.  If instantiation or a health-check raises, it logs a warning and
        falls through to the next adapter.  If all fail the last exception is
        re-raised.

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

        # Try adapters in priority order with fallback
        last_exc: Exception | None = None
        for adapter_name, adapter in adapters:
            try:
                # Quick validation: if the adapter is callable, just return it
                return adapter
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Adapter '%s' for %s failed, trying next: %s",
                    adapter_name,
                    capability.__name__,
                    exc,
                )

        # All adapters failed — should not reach here since return is inside try,
        # but kept for safety.
        if last_exc is not None:
            raise last_exc
        raise KeyError(f"No adapter registered for capability {capability.__name__}")

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

    def get_all(self, capability: ProviderType) -> list[tuple[str, Any]]:
        """Get all adapters for a capability in priority order."""
        return list(self._adapters.get(capability, []))

    def capabilities(self) -> list[ProviderType]:
        """List all registered capabilities."""
        return list(self._adapters.keys())


def _load_config_lazy() -> Any:
    """Lazy import of load_config to avoid circular imports at module level."""
    from qracer.config.loader import load_config

    return load_config()


def build_registry() -> DataRegistry:
    """Build a DataRegistry from providers.toml configuration.

    Reads the providers config, instantiates each enabled built-in adapter,
    and registers it with its capabilities.  Providers are registered in
    priority order (lower number = higher priority).
    """
    import importlib

    config = _load_config_lazy()
    registry = DataRegistry()

    # Sort providers by priority (lower = higher priority)
    sorted_providers = sorted(
        config.providers.providers.items(),
        key=lambda item: item[1].priority,
    )

    for name, prov_cfg in sorted_providers:
        if not prov_cfg.enabled:
            logger.debug("Skipping disabled provider: %s", name)
            continue

        if name not in _BUILTIN_PROVIDERS:
            logger.warning("Unknown provider '%s' in config, skipping", name)
            continue

        adapter_path, cap_paths = _BUILTIN_PROVIDERS[name]

        try:
            # Import adapter class
            module_path, class_name = adapter_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            adapter_cls = getattr(module, class_name)
            adapter = adapter_cls()

            # Import capability classes
            caps: list[ProviderType] = []
            for cap_path in cap_paths:
                cap_mod_path, cap_name = cap_path.rsplit(".", 1)
                cap_mod = importlib.import_module(cap_mod_path)
                caps.append(getattr(cap_mod, cap_name))

            registry.register(name, adapter, caps)
            logger.info("Registered provider: %s", name)
        except Exception:
            logger.warning("Failed to instantiate provider '%s'", name, exc_info=True)

    return registry
