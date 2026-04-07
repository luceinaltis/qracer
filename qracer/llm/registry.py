"""LLMRegistry — role-based routing with per-role overrides.

Adapters register for roles. Agents request completions by role, not by provider.
Per-role assignment is overridable via config.
"""

from __future__ import annotations

from qracer.llm.providers import LLMProvider, Role


class LLMRegistry:
    """Routes LLM requests by role with ordered fallback."""

    def __init__(self) -> None:
        # role -> list of (name, provider) in priority order
        self._providers: dict[Role, list[tuple[str, LLMProvider]]] = {}

    def register(self, name: str, provider: LLMProvider, roles: list[Role]) -> None:
        """Register a provider for the given roles.

        Providers registered first for a given role have higher priority.
        """
        for role in roles:
            if role not in self._providers:
                self._providers[role] = []
            self._providers[role].append((name, provider))

    def get(self, role: Role, name: str | None = None) -> LLMProvider:
        """Get a provider by role, optionally by explicit name.

        Args:
            role: The role to look up.
            name: If provided, return the provider with this name specifically.

        Returns:
            The provider instance.

        Raises:
            KeyError: If no provider is registered for the role (or name).
        """
        providers = self._providers.get(role)
        if not providers:
            raise KeyError(f"No provider registered for role {role.value}")

        if name is not None:
            for provider_name, provider in providers:
                if provider_name == name:
                    return provider
            raise KeyError(f"No provider named '{name}' for role {role.value}")

        return providers[0][1]

    def get_all(self, role: Role) -> list[tuple[str, LLMProvider]]:
        """Get all providers for a role in priority order."""
        return list(self._providers.get(role, []))

    def roles(self) -> list[Role]:
        """List all registered roles."""
        return list(self._providers.keys())
