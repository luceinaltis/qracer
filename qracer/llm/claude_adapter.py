"""ClaudeAdapter — LLM provider for Anthropic Claude models.

Maps Tracer roles to appropriate Claude models:
- researcher → Sonnet (fast, cost-effective for data gathering)
- analyst    → Opus (deep analysis)
- strategist → Opus (investment decisions)
- reporter   → Haiku (summaries, low cost)
"""

from __future__ import annotations

from qracer.llm.providers import (
    CompletionRequest,
    CompletionResponse,
    Role,
)

try:
    import anthropic  # pyright: ignore[reportMissingImports]

    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

# Default role-to-model mapping
DEFAULT_MODEL_MAP: dict[Role, str] = {
    Role.RESEARCHER: "claude-sonnet-4-20250514",
    Role.ANALYST: "claude-opus-4-20250514",
    Role.STRATEGIST: "claude-opus-4-20250514",
    Role.REPORTER: "claude-haiku-4-20250514",
}

# Approximate per-token costs (USD) — input / output per 1M tokens
_MODEL_COSTS: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-opus-4-20250514": (15.0, 75.0),
    "claude-haiku-4-20250514": (0.25, 1.25),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a completion."""
    input_rate, output_rate = _MODEL_COSTS.get(model, (0.0, 0.0))
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


class ClaudeAdapter:
    """LLM adapter for Anthropic Claude models."""

    roles: list[Role] = list(Role)

    def __init__(
        self,
        api_key: str | None = None,
        model_map: dict[Role, str] | None = None,
    ) -> None:
        if not _HAS_ANTHROPIC:
            raise ImportError("anthropic is not installed. Install it with: uv add anthropic")
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model_map = model_map or dict(DEFAULT_MODEL_MAP)

    def model_for_role(self, role: Role) -> str:
        """Get the model assigned to a role."""
        return self._model_map.get(role, DEFAULT_MODEL_MAP[Role.RESEARCHER])

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send a completion request to Claude.

        Args:
            request: The completion request.
                     If request.model is not set, defaults to the RESEARCHER model.
        """
        model = request.model or self.model_for_role(Role.RESEARCHER)

        # Separate system message from conversation messages
        system_text: str | None = None
        messages: list[anthropic.types.MessageParam] = []
        for msg in request.messages:
            if msg.role == "system":
                system_text = msg.content
            else:
                messages.append({"role": msg.role, "content": msg.content})  # type: ignore[typeddict-item]

        kwargs: dict[str, object] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": messages,
        }
        if system_text is not None:
            kwargs["system"] = system_text

        response = await self._client.messages.create(**kwargs)  # type: ignore[arg-type]

        content = ""
        if response.content:
            block = response.content[0]
            if hasattr(block, "text"):
                content = block.text  # type: ignore[union-attr]
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = _estimate_cost(model, input_tokens, output_tokens)

        return CompletionResponse(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )
