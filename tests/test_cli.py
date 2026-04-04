"""Tests for CLI entrypoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tracer.cli import _build_registries, main


class TestBuildRegistries:
    def test_returns_registry_tuple(self) -> None:
        """Should return (LLMRegistry, DataRegistry) without crashing."""
        llm, data = _build_registries()
        # At minimum, registries are returned (may or may not have adapters
        # depending on installed packages and env vars).
        assert llm is not None
        assert data is not None


class TestMain:
    def test_main_runs_asyncio(self) -> None:
        """main() should call asyncio.run with the REPL."""
        with (
            patch("tracer.cli.asyncio") as mock_asyncio,
            patch("tracer.cli._build_registries") as mock_build,
        ):
            mock_build.return_value = (MagicMock(), MagicMock())
            main()
            mock_asyncio.run.assert_called_once()
