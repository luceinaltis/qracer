"""Base agent class — shared scaffolding for all Tracer pipeline agents."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from tracer.data.registry import DataRegistry
from tracer.llm.providers import CompletionRequest, CompletionResponse, Message, Role
from tracer.llm.registry import LLMRegistry
from tracer.models import ToolResult

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for pipeline agents.

    Each agent owns a single LLM role and uses the registries to fetch data
    and request completions.
    """

    role: Role  # subclasses set this as a class variable

    def __init__(self, llm: LLMRegistry, data: DataRegistry) -> None:
        self._llm = llm
        self._data = data

    async def _complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> CompletionResponse:
        """Request an LLM completion using this agent's role."""
        provider = self._llm.get(self.role)
        request = CompletionRequest(
            messages=[
                Message(role="system", content=system),
                Message(role="user", content=user),
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return await provider.complete(request)

    @staticmethod
    def _successful_results(results: list[ToolResult]) -> list[ToolResult]:
        """Filter to only successful tool results."""
        return [r for r in results if r.success]

    @staticmethod
    def _format_tool_data(results: list[ToolResult]) -> str:
        """Serialize successful tool results into a prompt-friendly string."""
        sections: list[str] = []
        for r in results:
            if r.success:
                sections.append(
                    f"[{r.tool}] source={r.source} stale={r.is_stale}\n"
                    f"{json.dumps(r.data, indent=2)}"
                )
        return "\n\n".join(sections)

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's pipeline step."""
