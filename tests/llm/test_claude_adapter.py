"""Tests for ClaudeAdapter interface (no live API calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from qracer.llm.providers import Role


class TestClaudeAdapterImport:
    def test_import_error_without_anthropic(self) -> None:
        """ClaudeAdapter raises ImportError if anthropic is not installed."""
        with patch.dict("sys.modules", {"anthropic": None}):
            # Re-import to trigger the ImportError path
            import importlib

            import qracer.llm.claude_adapter as mod

            importlib.reload(mod)
            with pytest.raises(ImportError, match="anthropic is not installed"):
                mod.ClaudeAdapter()


class TestClaudeAdapterModelMapping:
    def test_default_model_map(self) -> None:
        from qracer.llm.claude_adapter import DEFAULT_MODEL_MAP

        assert "sonnet" in DEFAULT_MODEL_MAP[Role.RESEARCHER]
        assert "opus" in DEFAULT_MODEL_MAP[Role.ANALYST]
        assert "opus" in DEFAULT_MODEL_MAP[Role.STRATEGIST]
        assert "haiku" in DEFAULT_MODEL_MAP[Role.REPORTER]

    def test_cost_estimation(self) -> None:
        from qracer.llm.claude_adapter import _estimate_cost

        cost = _estimate_cost("claude-sonnet-4-20250514", 1000, 500)
        # 1000 * 3.0 / 1M + 500 * 15.0 / 1M = 0.003 + 0.0075 = 0.0105
        assert abs(cost - 0.0105) < 1e-6

    def test_cost_estimation_unknown_model(self) -> None:
        from qracer.llm.claude_adapter import _estimate_cost

        assert _estimate_cost("unknown-model", 1000, 500) == 0.0


class TestClaudeAdapterComplete:
    @pytest.fixture()
    def mock_anthropic(self) -> MagicMock:
        """Create a mock anthropic module and client."""
        mock_mod = MagicMock()
        mock_mod.NOT_GIVEN = object()
        return mock_mod

    def test_model_for_role(self, mock_anthropic: MagicMock) -> None:
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            import importlib

            import qracer.llm.claude_adapter as mod

            importlib.reload(mod)
            adapter = mod.ClaudeAdapter(api_key="test-key")
            assert "sonnet" in adapter.model_for_role(Role.RESEARCHER)
            assert "opus" in adapter.model_for_role(Role.ANALYST)

    def test_custom_model_map(self, mock_anthropic: MagicMock) -> None:
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            import importlib

            import qracer.llm.claude_adapter as mod

            importlib.reload(mod)
            custom = {role: "custom-model" for role in Role}
            adapter = mod.ClaudeAdapter(api_key="test-key", model_map=custom)
            assert adapter.model_for_role(Role.RESEARCHER) == "custom-model"
