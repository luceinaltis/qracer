"""Tests for LLM provider protocols and data classes."""

from __future__ import annotations

from tracer.llm.providers import (
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
    Message,
    Role,
)


class TestMessage:
    def test_create(self) -> None:
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_frozen(self) -> None:
        msg = Message(role="user", content="Hello")
        try:
            msg.role = "system"  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass


class TestCompletionRequest:
    def test_defaults(self) -> None:
        req = CompletionRequest(messages=[Message(role="user", content="Hi")])
        assert req.model is None
        assert req.max_tokens == 4096
        assert req.temperature == 0.0
        assert req.stop_sequences == []

    def test_custom_values(self) -> None:
        req = CompletionRequest(
            messages=[Message(role="user", content="Hi")],
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            temperature=0.7,
            stop_sequences=["STOP"],
        )
        assert req.model == "claude-sonnet-4-20250514"
        assert req.max_tokens == 1024
        assert req.temperature == 0.7
        assert req.stop_sequences == ["STOP"]


class TestCompletionResponse:
    def test_create(self) -> None:
        resp = CompletionResponse(
            content="Hello!",
            model="claude-sonnet-4-20250514",
            input_tokens=10,
            output_tokens=5,
            cost=0.0001,
        )
        assert resp.content == "Hello!"
        assert resp.model == "claude-sonnet-4-20250514"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.cost == 0.0001


class TestRole:
    def test_values(self) -> None:
        assert Role.RESEARCHER.value == "researcher"
        assert Role.ANALYST.value == "analyst"
        assert Role.STRATEGIST.value == "strategist"
        assert Role.REPORTER.value == "reporter"

    def test_all_roles(self) -> None:
        assert len(Role) == 4


class TestProtocolConformance:
    def test_llm_provider_protocol(self) -> None:
        """Verify that a class with the right methods satisfies LLMProvider."""

        class MyProvider:
            async def complete(self, request: CompletionRequest) -> CompletionResponse:
                return CompletionResponse(
                    content="test",
                    model="test-model",
                    input_tokens=0,
                    output_tokens=0,
                    cost=0.0,
                )

        assert isinstance(MyProvider(), LLMProvider)
