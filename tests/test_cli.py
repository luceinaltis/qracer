"""Tests for CLI internal helpers."""

from __future__ import annotations

from tracer.cli import _build_registries


class TestBuildRegistries:
    def test_returns_registry_tuple(self) -> None:
        """Should return (LLMRegistry, DataRegistry) without crashing."""
        llm, data = _build_registries()
        assert llm is not None
        assert data is not None
