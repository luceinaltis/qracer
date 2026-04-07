"""LLM layer protocols.

Each LLM provider implements the LLMProvider protocol.
The LLMRegistry routes completion requests by role with fallback support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


class Role(Enum):
    """LLM roles used by the Tracer pipeline."""

    RESEARCHER = "researcher"
    ANALYST = "analyst"
    STRATEGIST = "strategist"
    REPORTER = "reporter"


@dataclass(frozen=True)
class Message:
    """A single message in a completion request."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass(frozen=True)
class CompletionRequest:
    """Request for an LLM completion."""

    messages: list[Message]
    model: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    stop_sequences: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CompletionResponse:
    """Response from an LLM completion."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float


@runtime_checkable
class LLMProvider(Protocol):
    """Capability: LLM completion."""

    async def complete(self, request: CompletionRequest) -> CompletionResponse: ...
