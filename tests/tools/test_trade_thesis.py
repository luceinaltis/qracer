"""Tests for trade thesis generation (pipeline step 7)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from helpers import make_single_role_registry, sample_analysis_results

from tracer.llm.providers import Role
from tracer.llm.registry import LLMRegistry
from tracer.models import TradeThesis
from tracer.tools.pipeline import trade_thesis

_VALID_LLM_RESPONSE = json.dumps(
    {
        "entry_zone": [180.0, 185.0],
        "target_price": 210.0,
        "stop_loss": 170.0,
        "catalyst": "AI services revenue growth",
        "catalyst_date": "Q2 2026",
        "conviction": 8,
        "summary": "AAPL is well-positioned for upside driven by AI services revenue.",
    }
)


class TestTradeThesis:
    async def test_success(self) -> None:
        registry = make_single_role_registry(Role.STRATEGIST, _VALID_LLM_RESPONSE)
        result = await trade_thesis("AAPL", sample_analysis_results(), registry)

        assert result.success is True
        assert result.tool == "trade_thesis"
        thesis = result.data["thesis"]
        assert thesis["ticker"] == "AAPL"
        assert thesis["entry_zone"] == [180.0, 185.0]
        assert thesis["target_price"] == 210.0
        assert thesis["stop_loss"] == 170.0
        assert thesis["conviction"] == 8
        assert thesis["catalyst"] == "AI services revenue growth"
        assert thesis["catalyst_date"] == "Q2 2026"
        assert isinstance(thesis["summary"], str)

    async def test_risk_reward_ratio_computed(self) -> None:
        registry = make_single_role_registry(Role.STRATEGIST, _VALID_LLM_RESPONSE)
        result = await trade_thesis("AAPL", sample_analysis_results(), registry)

        assert result.success is True
        thesis = result.data["thesis"]
        # entry_mid = (180 + 185) / 2 = 182.5
        # rr = (210 - 182.5) / (182.5 - 170) = 27.5 / 12.5 = 2.2
        assert thesis["risk_reward_ratio"] == 2.2

    async def test_conviction_range_validation(self) -> None:
        """Conviction outside 1-10 should cause failure."""
        bad_response = json.dumps(
            {
                "entry_zone": [180.0, 185.0],
                "target_price": 210.0,
                "stop_loss": 170.0,
                "catalyst": "test",
                "catalyst_date": None,
                "conviction": 15,
                "summary": "test",
            }
        )
        registry = make_single_role_registry(Role.STRATEGIST, bad_response)
        result = await trade_thesis("AAPL", sample_analysis_results(), registry)

        assert result.success is False
        assert result.error is not None

    async def test_invalid_json_response(self) -> None:
        """Non-JSON LLM response should return failure."""
        registry = make_single_role_registry(Role.STRATEGIST, "This is not JSON at all")
        result = await trade_thesis("AAPL", sample_analysis_results(), registry)

        assert result.success is False
        assert "Failed to parse" in (result.error or "")

    async def test_missing_keys_in_response(self) -> None:
        """Missing required keys should return failure."""
        registry = make_single_role_registry(
            Role.STRATEGIST, json.dumps({"entry_zone": [180, 185]})
        )
        result = await trade_thesis("AAPL", sample_analysis_results(), registry)

        assert result.success is False

    async def test_llm_provider_error(self) -> None:
        """LLM provider raising an exception should return failure."""
        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = RuntimeError("LLM unavailable")
        registry = LLMRegistry()
        registry.register("mock", mock_provider, [Role.STRATEGIST])

        result = await trade_thesis("AAPL", sample_analysis_results(), registry)

        assert result.success is False
        assert result.error is not None

    async def test_catalyst_date_nullable(self) -> None:
        """catalyst_date can be null."""
        response = json.dumps(
            {
                "entry_zone": [180.0, 185.0],
                "target_price": 210.0,
                "stop_loss": 170.0,
                "catalyst": "General growth",
                "catalyst_date": None,
                "conviction": 5,
                "summary": "Moderate outlook.",
            }
        )
        registry = make_single_role_registry(Role.STRATEGIST, response)
        result = await trade_thesis("AAPL", sample_analysis_results(), registry)

        assert result.success is True
        assert result.data["thesis"]["catalyst_date"] is None

    async def test_empty_analysis_results(self) -> None:
        """Should work even with no prior analysis results."""
        registry = make_single_role_registry(Role.STRATEGIST, _VALID_LLM_RESPONSE)
        result = await trade_thesis("AAPL", [], registry)

        assert result.success is True


class TestTradeThesisModel:
    def test_valid_conviction(self) -> None:
        thesis = TradeThesis(
            ticker="AAPL",
            entry_zone=(180.0, 185.0),
            target_price=210.0,
            stop_loss=170.0,
            risk_reward_ratio=2.2,
            catalyst="test",
            catalyst_date=None,
            conviction=5,
            summary="test",
        )
        assert thesis.conviction == 5

    def test_conviction_too_low(self) -> None:
        with pytest.raises(ValueError, match="Conviction must be between 1 and 10"):
            TradeThesis(
                ticker="AAPL",
                entry_zone=(180.0, 185.0),
                target_price=210.0,
                stop_loss=170.0,
                risk_reward_ratio=2.2,
                catalyst="test",
                catalyst_date=None,
                conviction=0,
                summary="test",
            )

    def test_conviction_too_high(self) -> None:
        with pytest.raises(ValueError, match="Conviction must be between 1 and 10"):
            TradeThesis(
                ticker="AAPL",
                entry_zone=(180.0, 185.0),
                target_price=210.0,
                stop_loss=170.0,
                risk_reward_ratio=2.2,
                catalyst="test",
                catalyst_date=None,
                conviction=11,
                summary="test",
            )
